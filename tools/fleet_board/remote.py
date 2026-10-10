"""Read-only probe that runs ON a GPU box (python3 >= 3.8, stdlib only).

collect.py sends this file over ssh stdin; argv[1] is a JSON object
{"globs": [...], "window_min": 45} and optionally "patterns", "offsets" ({train.log path: byte offset}), "max_lines". Prints one compact JSON object. Nothing is modified.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time


def sh(cmd, timeout=15):
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             universal_newlines=True, timeout=timeout)
        return out.stdout
    except Exception:
        return ""


def cpu_times():
    rows = []
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3:4].isdigit():
                v = [int(x) for x in line.split()[1:]]
                rows.append((v[3] + v[4], sum(v)))  # idle+iowait, total
    return rows


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def parse_gpus():
    q = "pci.bus_id,utilization.gpu,memory.used,memory.total,power.draw,name"
    gpus = {}
    for line in sh(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"]).splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6:
            continue
        bus = p[0].split(":")[1].upper() if p[0].count(":") >= 2 else p[0].upper()
        gpus[bus] = {"bus": bus, "util": num(p[1]), "mem_mb": num(p[2]), "mem_total_mb": num(p[3]),
                     "power_w": num(p[4]), "name": p[5]}
    return gpus


def parse_apps():
    apps = {}
    for line in sh(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory",
                    "--format=csv,noheader,nounits"]).splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 2 and p[0].isdigit():
            apps[int(p[0])] = p[1].split(":")[1].upper()
    return apps


def procs():
    table = {}
    for line in sh(["ps", "-eo", "pid=,ppid=,args="]).splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = (int(parts[1]), parts[2] if len(parts) > 2 else "")
    return table


def run_of(cmdline):
    if "mtg_ml.rl.train" not in cmdline:
        return None
    t = cmdline.split()
    if "--run" in t and t.index("--run") + 1 < len(t):
        return os.path.realpath(t[t.index("--run") + 1])
    return None


def flag(cmdline, name):
    t = cmdline.split()
    return t[t.index(name) + 1] if name in t and t.index(name) + 1 < len(t) else None


def allowed(pid):
    try:
        with open("/proc/%d/status" % pid) as f:
            for line in f:
                if line.startswith("Cpus_allowed_list:"):
                    out = set()
                    for part in line.split(":", 1)[1].strip().split(","):
                        a, _, b = part.partition("-")
                        out.update(range(int(a), int(b or a) + 1))
                    return out
    except Exception:
        pass
    return set()


def ranges(cpus):
    cpus = sorted(cpus)
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append([cpus[i], cpus[j]])
        i = j + 1
    return out


def jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def isnum(x):
    return isinstance(x, (int, float)) and x == x


def metrics(path, window_min):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    rows.append(r)
    except OSError:
        return {}
    if not rows:
        return {}
    last = rows[-1]
    keep = ("games_total", "win_vs_frozen", "update_s", "wall_s", "rollout_s", "lr", "elapsed_s")
    out = {k: last.get(k) for k in keep if isnum(last.get(k))}
    wf = [r["win_vs_frozen"] for r in rows if isnum(r.get("win_vs_frozen"))]
    if wf:
        out["win_vs_frozen"] = sum(wf[-5:]) / len(wf[-5:])
        out["win_vs_frozen_first"] = sum(wf[:5]) / len(wf[:5])
    ev = []
    for r in rows:
        if any(k in r for k in ("ladder/elo", "bench/jund_vs_bot", "bench/jund_vs_bot_greedy")):
            ev.append({"games": r.get("games_total"), "bench": r.get("bench/jund_vs_bot"),
                       "greedy": r.get("bench/jund_vs_bot_greedy"), "elo": r.get("ladder/elo"),
                       "se": r.get("ladder/elo_se")})
    out["evals"] = ev
    # games/h over the trailing window: prefer elapsed_s, else summed wall_s.
    g_end, window = last.get("games_total"), window_min * 60.0
    if isnum(g_end):
        if isnum(last.get("elapsed_s")):
            t_end, ref = last["elapsed_s"], None
            for r in reversed(rows):
                if isnum(r.get("elapsed_s")) and isnum(r.get("games_total")):
                    ref = r
                    if t_end - r["elapsed_s"] >= window:
                        break
            if ref is not None and t_end - ref["elapsed_s"] > 60:
                out["games_per_h"] = (g_end - ref["games_total"]) / (t_end - ref["elapsed_s"]) * 3600
        else:
            acc, ref = 0.0, None
            for r in reversed(rows):
                if not (isnum(r.get("wall_s")) and isnum(r.get("games_total"))):
                    continue
                acc += r["wall_s"]
                ref = r
                if acc >= window:
                    break
            if ref is not None and acc > 60:
                out["games_per_h"] = (g_end - ref["games_total"] + (ref.get("games") or 0)) / acc * 3600
    try:
        out["age_s"] = time.time() - os.path.getmtime(path)
    except OSError:
        pass
    return out


def scan_log(path, offsets, patterns, max_lines, max_bytes=524288):
    """Read-only: size and age of train.log, plus error lines added since the stored byte offset.

    No stored offset (first look) only records the size. At most max_lines short lines come back."""
    out = {}
    try:
        st = os.stat(path)
    except OSError:
        return out
    out["log_size"], out["log_age_s"] = st.st_size, time.time() - st.st_mtime
    off = offsets.get(path) if offsets else None
    if not patterns or off is None:
        return out
    if off > st.st_size:  # truncated or rotated: look at the start again
        off = 0
    try:
        with open(path, "rb") as f:
            f.seek(max(off, st.st_size - max_bytes))
            text = f.read(max_bytes).decode("utf-8", "replace")
    except OSError:
        return out
    rx = re.compile("|".join("(?:%s)" % p for p in patterns))
    hits, tb = [], False
    for line in text.splitlines():
        if tb and line.strip() and not line[:1].isspace() and not line.startswith("Traceback"):
            hits[-1] = "Traceback: " + line.strip()  # the exception line closes the traceback
            tb = False
        elif rx.search(line):
            hits.append(line.strip())
            tb = line.startswith("Traceback")
    out["errors"] = [h[:160] for h in hits[-max_lines:]]
    out["error_count"] = len(hits)
    return out


def main():
    cfg = json.loads(sys.argv[1])
    t0 = cpu_times()
    s1 = parse_gpus()
    time.sleep(0.4)
    s2 = parse_gpus()
    time.sleep(0.4)
    t1 = cpu_times()
    s3 = parse_gpus()
    percpu = [100.0 * (1 - (i1 - i0) / (a1 - a0)) if a1 > a0 else 0.0
              for (i0, a0), (i1, a1) in zip(t0, t1)]
    gpus = {}
    for bus in set(s1) | set(s2) | set(s3):
        samples = [s[bus] for s in (s1, s2, s3) if bus in s]
        g = dict(samples[-1])
        utils = [x["util"] for x in samples if x["util"] is not None]
        g["util_min"], g["util_max"] = (min(utils), max(utils)) if utils else (None, None)
        gpus[bus] = g
    apps = parse_apps()
    table = procs()

    def owner(pid):
        hops = 0
        while pid in table and hops < 12:
            run = run_of(table[pid][1])
            if run:
                return pid, run
            pid, hops = table[pid][0], hops + 1
        return None, None

    live = {}
    for pid, (_ppid, cmd) in table.items():
        r = run_of(cmd)
        if r:
            live[r] = {"pid": pid, "cmd": cmd}
    runs = {}
    for pid, bus in apps.items():
        lp, run = owner(pid)
        if run is None:
            if bus in gpus:
                try:
                    with open("/proc/%d/comm" % pid) as f:
                        gpus[bus].setdefault("foreign", f.read().strip())
                except OSError:
                    gpus[bus].setdefault("foreign", "other process")
            continue
        info = runs.setdefault(run, {"learner_bus": None, "server_buses": []})
        if lp == pid:
            info["learner_bus"] = bus
        elif bus not in info["server_buses"]:
            info["server_buses"].append(bus)
    # CPU envelope per run: union of affinities over its process tree, ignoring full-mask processes.
    ncpu = os.cpu_count() or 1
    children = {}
    for pid, (ppid, _c) in table.items():
        children.setdefault(ppid, []).append(pid)
    for run, lv in live.items():
        stack, cpus = [lv["pid"]], set()
        while stack:
            p = stack.pop()
            stack.extend(children.get(p, []))
            a = allowed(p)
            if a and len(a) < ncpu:
                cpus |= a
        runs.setdefault(run, {"learner_bus": None, "server_buses": []})["cpus"] = ranges(cpus)

    dirs, seen = [], set()
    for g in cfg["globs"]:
        for d in sorted(glob.glob(os.path.expanduser(g["glob"]))):
            if os.path.isdir(d) and any(os.path.exists(os.path.join(d, n)) for n in ("metrics.jsonl", "process.json")):
                dirs.append((d, g.get("color")))
                seen.add(os.path.realpath(d))
    for d in list(runs):
        if d not in seen and os.path.isdir(d):
            dirs.append((d, None))
            seen.add(d)
    out_runs = []
    for d, color in dirs:
        real = os.path.realpath(d)
        entry = {"dir": d, "name": os.path.basename(d.rstrip("/")), "color": color}
        proc = jload(os.path.join(d, "process.json")) or {}
        entry["status"] = proc.get("status")
        camp = jload(os.path.join(os.path.dirname(d.rstrip("/")), "campaign.json"))
        if isinstance(camp, dict):
            camp = [camp]
        for c in camp or []:
            if isinstance(c, dict) and os.path.realpath(c.get("run", "")) == real:
                entry["handle"] = c.get("handle")
                entry["target_games"] = c.get("target_games")
        cmd = live.get(real, {}).get("cmd", "")
        if cmd:
            tg = flag(cmd, "--total-games")
            if tg and tg.isdigit() and not entry.get("target_games"):
                entry["target_games"] = int(tg)
            entry["hidden"] = flag(cmd, "--hidden")
            entry["lr"] = flag(cmd, "--ppo-lr")
        entry["live"] = real in live
        entry.update(runs.get(real, {}))
        entry["metrics"] = metrics(os.path.join(d, "metrics.jsonl"), cfg.get("window_min", 45))
        entry.update(scan_log(os.path.join(d, "train.log"), cfg.get("offsets"), cfg.get("patterns"),
                              cfg.get("max_lines", 3)))
        out_runs.append(entry)
    print(json.dumps({"nproc": ncpu,
                      "cpu_busy": round(sum(percpu) / len(percpu), 1) if percpu else None,
                      "gpus": sorted(gpus.values(), key=lambda g: g["bus"]), "runs": out_runs}))


if __name__ == "__main__":  # `python3 -` over ssh runs as __main__
    main()
