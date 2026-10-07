"""Fetch metrics + health of the next-actions fine-tune arms from h100-private; one JSON doc per run in out/.
Per-iteration rows binned (BIN iterations each); every evaluation row kept."""
import json, pathlib, subprocess, time

OUT = pathlib.Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
BIN = 25
ARMS = {  # run id: (label, what changes vs the control, order, kind)
    "r1-control": ("Control", "Overnight checkpoint + 1M games with the fixed engine (attacker trap) and the value clamp; nothing else changed.", 0, "ft"),
    "r1-lranneal": ("LR anneal", "lr 3e-4 to 3e-5 linearly over the 1M games (item 7).", 1, "ft"),
    "r1-gammaturn": ("Per-turn discount", "gamma 0.97 and lambda 0.95 per game turn instead of 0.995 / 0.95 per decision (item 8).", 2, "ft"),
    "r2-features": ("New features", "Readiness / lethal features, skip_untap, stack targets and X, known library positions, option previews (items 10-12).", 3, "ft"),
    "r2-botjund": ("Bot games as Jund", "With the new features (all round-2 arms have them; compare to New features). 25% of games learner Jund vs the blue bot, taken from the pool share (item 15).", 4, "ft"),
    "r2-pfsp": ("PFSP pool", "With the new features (compare to New features). Pool opponents weighted by (1 - win rate)^2 instead of uniform (item 14).", 5, "ft"),
    "r2-automana": ("Auto mana", "With the new features (compare to New features). Mana paid automatically except Spawn sacrifices; Spawn-only priority stops skipped (item 17).", 6, "ft"),
    "r3-control": ("R3 control", "Round 3 parent: the LR-anneal final checkpoint (9.4M games), 1M more games at lr 3e-5, with the new features (on in this code for all round 2-3 arms). Round-3 arms compare to this.", 7, "ft"),
    "r3-attn": ("R3 entity attention", "From the LR-anneal final: 1 self-attention layer over entities, initialised as identity, fresh optimizer, lr 3e-5 (item 17).", 8, "ft"),
    "r3-postboard": ("R3 postboard 0.2", "From the LR-anneal final: 20% sideboarded games instead of 50%, lr 3e-5 (item 13).", 10, "ft"),
    "r3-botjund": ("R3 bot games as Jund", "From the LR-anneal final: 25% of games learner Jund vs the blue bot, lr 3e-5 (item 15).", 11, "ft"),
    "r1-exploit-jund": ("Exploiter · Jund", "Learner on Jund vs the frozen main policy only, 150 iterations (item 14 test).", 20, "exploit"),
    "r1-exploit-blue": ("Exploiter · Blue", "Learner on Blue vs the frozen main policy only, 150 iterations (item 14 test).", 21, "exploit"),
}
DIRS = ["~/mtg-ml-next/runs", "~/mtg-ml-next2/runs"]
KEEP = ("win_vs_pool", "win_vs_main", "win_vs_bot", "entropy", "nt_entropy", "approx_kl", "explained_var", "lr", "decisions_per_game", "wall_s")

cmd = []
for rid in ARMS:
    for base in DIRS:
        d = f"{base}/{rid}"
        cmd.append(f'if [ -d {d} ]; then echo "@@RUN {rid}"; cat {d}/metrics.jsonl 2>/dev/null; echo "@@HEALTH {rid}"; '
                   f'p=$(pgrep -f "mtg_ml.rl.train --run [r]uns/{rid} " | head -1); echo "pid=$p"; [ -n "$p" ] && echo "uptime=$(ps -o etimes= -p $p | tr -d " ")"; '
                   f'stat -c "mtime=%Y" {d}/metrics.jsonl 2>/dev/null; echo "now=$(date +%s)"; '
                   f'echo "errtail=$(grep -E -A30 "Traceback" {d}.log 2>/dev/null | grep -E "Error|error" | tail -1 | cut -c1-300)"; fi')
cmd.append('echo "@@HOST"; echo "load=$(cut -d" " -f1 /proc/loadavg)"; echo "disk=$(df -BG --output=avail ~ | tail -1 | tr -d " ")"')
raw = subprocess.run(["ssh", "-o", "ConnectTimeout=20", "h100-private", "; ".join(cmd)], capture_output=True, text=True, timeout=180).stdout
sec, cur = {}, None
for line in raw.splitlines():
    if line.startswith("@@"):
        cur = line[2:]
        sec[cur] = []
    elif cur:
        sec[cur].append(line)
host = dict(x.split("=", 1) for x in sec.get("HOST", []) if "=" in x)
now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
for rid, (label, desc, order, kind) in ARMS.items():
    if f"RUN {rid}" not in sec:
        continue
    seen = {}
    for line in sec[f"RUN {rid}"]:
        try:
            r = json.loads(line)
            seen[r["iteration"]] = r
        except Exception:
            pass
    rows = [seen[k] for k in sorted(seen)]
    bins, evals = [], []
    for i in range(0, len(rows), BIN):
        chunk = rows[i:i + BIN]
        b = {"games_total": chunk[-1]["games_total"], "iteration": chunk[-1]["iteration"]}
        for k in KEEP:
            xs = [r[k] for r in chunk if isinstance(r.get(k), (int, float))]
            if xs:
                b[k] = round(sum(xs) / len(xs), 5)
        bins.append(b)
    for r in rows:
        if "bench/jund_vs_bot" in r or "ladder/elo" in r:
            evals.append({"games_total": r["games_total"], "iteration": r["iteration"],
                          "sampled": r.get("bench/jund_vs_bot"), "sampled_ci": r.get("bench/jund_vs_bot_ci"),
                          "greedy": r.get("bench/jund_vs_bot_greedy"), "greedy_ci": r.get("bench/jund_vs_bot_greedy_ci"),
                          "elo": r.get("ladder/elo"), "elo_se": r.get("ladder/elo_se")})
    h = dict(x.split("=", 1) for x in sec.get(f"HEALTH {rid}", []) if "=" in x)
    alive = bool(h.get("pid"))
    now, mtime = int(h.get("now", time.time())), int(h.get("mtime", 0) or 0)
    last = rows[-1] if rows else {}
    target = 307200 if kind == "exploit" else 10_400_000 if rid.startswith("r3-") and rows and rows[0]["games_total"] > 1e6 else 1_000_000 if rows and rows[0]["games_total"] < 1e6 else 9_400_000
    done = bool(last) and last.get("games_total", 0) >= target - 2048
    doc = {"ctrl": "r3-control" if rid.startswith("r3-") else "r2-features" if rid in ("r2-botjund", "r2-pfsp", "r2-automana") else "r1-control", "label": label, "desc": desc, "order": order, "kind": kind, "updated_at": now_iso,
           "status": "running" if alive else ("finished" if done else "down"),
           "health": {"uptime_s": int(h.get("uptime", 0) or 0), "last_iter_age_s": (now - mtime) if mtime else None,
                      "last_error": h.get("errtail", "").strip(), "host_load": float(host.get("load", 0) or 0), "disk_free": host.get("disk", "?")},
           "last": {k: last.get(k) for k in ("iteration", "games_total", "entropy", "win_vs_pool", "win_vs_main", "explained_var", "lr", "wall_s")},
           "start_games": rows[0]["games_total"] - 2048 if rows else None, "target_games": target,
           "bins": bins, "evals": evals}
    (OUT / f"{rid}.json").write_text(json.dumps(doc))
    print(rid, doc["status"], len(rows), "rows", len(evals), "evals", len(json.dumps(doc)), "bytes")
