"""Only artifact/witness validation is public in this foundation release."""

import argparse

from .jsonio import write_json
from .validation import validate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("validate")
    command.add_argument("--manifest", required=True)
    command.add_argument("--engines", default="python,native")
    command.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = validate(args.manifest, tuple(args.engines.split(",")))
    try:
        write_json(args.out, report)
    except (OSError, ValueError) as e:
        parser.exit(1, f"benchmark: {e}\n")
    if report["status"] != "complete":
        parser.exit(1, "benchmark: " + "; ".join(report["errors"]) + "\n")


if __name__ == "__main__":
    main()
