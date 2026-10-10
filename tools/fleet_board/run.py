"""Collect plus render: `python -m tools.fleet_board [--out PATH]`. Prints the output path and one summary line."""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from .collect import collect, write_snapshot
from .render import render, summary_line

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.fleet_board", description=__doc__)
    ap.add_argument("--config", type=Path, default=HERE / "fleet.toml")
    ap.add_argument("--out", type=Path, default=Path(".context/fleet.html"))
    ap.add_argument("--json", type=Path, default=None, help="snapshot path (default: next to --out, fleet.json)")
    ap.add_argument("--timeout", type=int, default=45, help="per-machine ssh timeout in seconds")
    ap.add_argument("--from-json", type=Path, default=None, help="skip collection, render this snapshot")
    args = ap.parse_args(argv)

    config = tomllib.loads(args.config.read_text())
    if args.from_json:
        import json
        snapshot = json.loads(args.from_json.read_text())
    else:
        snapshot = collect(config, timeout=args.timeout)
        write_snapshot(snapshot, args.json or args.out.with_name("fleet.json"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(snapshot, config))
    print(args.out)
    print(summary_line(snapshot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
