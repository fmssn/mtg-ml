"""Collect live data from every machine in fleet.toml (one read-only ssh call each, in parallel)."""
from __future__ import annotations

import json
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = HERE / "remote.py"


def probe(machine: dict, timeout: int, window_min: int) -> dict:
    """One ssh call. Returns the remote JSON plus status; never raises."""
    arg = json.dumps({"globs": machine.get("runs", []), "window_min": window_min})
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={min(timeout, 10)}",
           machine["ssh"], f"python3 - {shlex.quote(arg)}"]
    t0 = time.time()
    try:
        out = subprocess.run(cmd, input=REMOTE.read_text(), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "unreachable", "error": f"timeout after {timeout}s"}
    except OSError as exc:
        return {"status": "unreachable", "error": str(exc)}
    if out.returncode != 0 or not out.stdout.strip():
        err = (out.stderr.strip().splitlines() or ["ssh failed"])[-1]
        return {"status": "unreachable", "error": err[:200]}
    try:
        data = json.loads(out.stdout)
    except ValueError:
        return {"status": "unreachable", "error": "bad JSON from remote probe"}
    data["status"] = "ok"
    data["probe_s"] = round(time.time() - t0, 1)
    return data


def collect(config: dict, timeout: int = 45) -> dict:
    machines = config["machines"]
    window = int(config.get("window_minutes", 45))
    with ThreadPoolExecutor(max_workers=max(1, len(machines))) as pool:
        results = list(pool.map(lambda m: probe(m, timeout, window), machines))
    return {"collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "machines": {m["name"]: r for m, r in zip(machines, results)}}


def write_snapshot(snapshot: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=1))
