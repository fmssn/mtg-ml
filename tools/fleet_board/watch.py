"""Watchdog: `python -m tools.fleet_board.watch`. Refreshes the board and exits only when someone must act.

Exit 1 with one line per event (crash, error, stall, finished, unreachable, started), exit 0 after
--max-minutes without events. Read-only on the boxes. Details go to watch.log next to --out."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import tomllib
from pathlib import Path

from .collect import collect, write_snapshot
from .render import render, summary_line

HERE = Path(__file__).resolve().parent
MAX_LINES = 10
DEFAULTS = {
    "interval": 10, "max_minutes": 60, "stall_minutes": 20, "unreachable_checks": 2, "max_error_lines": 3,
    "error_patterns": ["Traceback", "OutOfMemory", "out of memory", "CUDA error", r"\bKilled\b", "NativeRulesError"],
    "crash_status": ["failed", "crashed"], "done_status": ["complete", "completed"],
    "exclude": [],
}


def watch_settings(config: dict) -> dict:
    return {**DEFAULTS, **config.get("watch", {})}


def watch_config(config: dict, ws: dict) -> dict:
    """Config for collection: [watch.runs] (machine -> globs) replaces a machine's run globs."""
    over = ws.get("runs", {})
    machines = [dict(m, runs=[{"glob": g} if isinstance(g, str) else g for g in over[m["name"]]])
                if m["name"] in over else m for m in config["machines"]]
    return {**config, "machines": machines}


def scan_request(state: dict, ws: dict) -> dict:
    offsets: dict[str, dict[str, int]] = {}
    for key, r in state.get("runs", {}).items():
        if r.get("log") is not None and r.get("offset") is not None:
            offsets.setdefault(key.split("/", 1)[0], {})[r["log"]] = r["offset"]
    return {"patterns": ws["error_patterns"], "max_lines": ws["max_error_lines"], "offsets": offsets}


def _short(line: str, n: int = 100) -> str:
    return line if len(line) <= n else line[: n - 1] + "..."


def detect(state: dict, snapshot: dict, ws: dict, report_starts: bool = False,
           stall_minutes: float | None = None) -> tuple[list[str], dict]:
    """Compare a snapshot with the stored state. Returns (event lines, new state). Pure.

    Events fire on transitions (crash, finished, started) or once per episode (stall, unreachable),
    so the same event is never reported twice. An empty state is a baseline: nothing is reported
    except stalls and error lines already visible."""
    stall_s = 60 * (ws["stall_minutes"] if stall_minutes is None else stall_minutes)
    first = not state
    old_runs, old_down = state.get("runs", {}), state.get("down", {})
    runs, down, events = {}, {}, []
    excl = [re.compile(x) for x in ws["exclude"]]
    for mname, m in snapshot["machines"].items():
        if m.get("status") != "ok":
            n = old_down.get(mname, {}).get("n", 0) + 1
            rep = old_down.get(mname, {}).get("reported", False)
            if n >= ws["unreachable_checks"] and not rep:
                events.append(f"unreachable {mname}: {n} checks in a row ({_short(m.get('error') or '?', 60)})")
                rep = True
            down[mname] = {"n": n, "reported": rep}
            for k, r in old_runs.items():  # keep state while the box is out of sight
                if k.startswith(mname + "/"):
                    runs[k] = r
            continue
        for r in m.get("runs", []):
            if any(x.search(r["name"]) for x in excl):
                continue
            key = f"{mname}/{r['dir']}"
            label = f"{mname} {r.get('handle') or r['name']}"
            prev = old_runs.get(key)
            status, live = r.get("status"), bool(r.get("live"))
            crashed = status in ws["crash_status"] or (status == "running" and not live)
            cur = {"status": status, "crashed": crashed, "log": r["dir"].rstrip("/") + "/train.log",
                   "offset": r.get("log_size", (prev or {}).get("offset")),
                   "last_error": (r.get("errors") or [(prev or {}).get("last_error")])[-1],
                   "stalled": (prev or {}).get("stalled", False)}
            crash_event = False
            if crashed and prev is not None and not prev.get("crashed"):
                why = f"status {status}" if status != "running" else "trainer process gone, status still running"
                err = f"; last error: {_short(cur['last_error'])}" if cur["last_error"] else ""
                events.append(f"crash {label}: {why}{err}")
                crash_event = True
            if status in ws["done_status"] and prev is not None and prev["status"] not in ws["done_status"]:
                g = (r.get("metrics") or {}).get("games_total")
                events.append(f"finished {label}" + (f": {int(g):,} games" if g else ""))
            if prev is None and not first and report_starts:
                events.append(f"started {label}: status {status}")
            if r.get("errors") and not crash_event:
                events.append(f"error {label}: " + " | ".join(_short(e, 90) for e in r["errors"]))
            ages = [a for a in (r.get("log_age_s"), (r.get("metrics") or {}).get("age_s")) if a is not None]
            stalled = status == "running" and live and bool(ages) and min(ages) > stall_s
            if stalled and not cur["stalled"]:
                events.append(f"stall {label}: no log or metrics write for {int(min(ages) // 60)} min")
            cur["stalled"] = stalled
            runs[key] = cur
    return events, {"runs": runs, "down": down}


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.fleet_board.watch", description=__doc__)
    ap.add_argument("--config", type=Path, default=HERE / "fleet.toml")
    ap.add_argument("--out", type=Path, default=Path(".context/fleet.html"))
    ap.add_argument("--json", type=Path, default=None, help="snapshot path (default: fleet.json next to --out)")
    ap.add_argument("--state", type=Path, default=None, help="event state (default: watch-state.json next to --out)")
    ap.add_argument("--interval", type=float, default=None, help="minutes between checks (default 10)")
    ap.add_argument("--max-minutes", type=float, default=None, help="exit 0 after this long without events (default 60)")
    ap.add_argument("--stall-minutes", type=float, default=None, help="no write for this long = stall (default 20)")
    ap.add_argument("--report-starts", action="store_true", help="also report new runs")
    ap.add_argument("--timeout", type=int, default=45, help="per-machine ssh timeout in seconds")
    args = ap.parse_args(argv)

    config = tomllib.loads(args.config.read_text())
    ws = watch_settings(config)
    interval = 60 * (args.interval if args.interval is not None else ws["interval"])
    max_s = 60 * (args.max_minutes if args.max_minutes is not None else ws["max_minutes"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.state or args.out.with_name("watch-state.json")
    logging.basicConfig(filename=args.out.with_name("watch.log"), level=logging.INFO,
                        format="%(asctime)s %(message)s")
    log = logging.getLogger("watch")
    cfg = watch_config(config, ws)
    state, t_end = load_state(state_path), time.time() + max_s
    while True:
        snapshot = collect(cfg, timeout=args.timeout, scan=scan_request(state, ws))
        write_snapshot(snapshot, args.json or args.out.with_name("fleet.json"))
        args.out.write_text(render(snapshot, config))
        events, state = detect(state, snapshot, ws, args.report_starts, args.stall_minutes)
        state_path.write_text(json.dumps(state, indent=1))
        summary = summary_line(snapshot)
        log.info("check: %s; %d events", summary, len(events))
        if events:
            for e in events:
                log.info("event: %s", e)
            shown = events[:MAX_LINES]
            if len(events) > MAX_LINES:
                shown.append(f"... and {len(events) - MAX_LINES} more events (see watch.log)")
            print("\n".join(shown))
            print(summary)
            return 1
        if time.time() + interval > t_end:
            print(summary)
            print(args.out)
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
