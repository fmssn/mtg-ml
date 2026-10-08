"""Bounded native build: failure is actionable review context, not a skipped model run."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from review_result import read_json
from runner import run_phase


def run_build(tmp, env):
    return run_phase(["maturin", "develop", "--release", "--locked", "--manifest-path", "native/Cargo.toml"],
                     "", 300, tmp / "native-build.log", env)


def build(tmp):
    tmp = Path(tmp)
    manifest = read_json(tmp / "manifest.json", {})
    unmerged = subprocess.run(("git", "diff", "--name-only", "--diff-filter=U", "-z"), capture_output=True, text=True, check=True).stdout
    conflicts = set((tmp / "conflicts.txt").read_text().splitlines()) if (tmp / "conflicts.txt").exists() else set()
    remaining = set(filter(None, unmerged.split("\0")))
    # The trusted controller stages only the declared, marker-free resolutions.
    # This lets the next build run without granting the model git-add permission.
    if remaining and remaining <= conflicts and all(Path(p).is_file() and not re.search(r"^(<<<<<<<|>>>>>>>) ", Path(p).read_text(errors="replace"), re.M) for p in remaining):
        subprocess.run(("git", "add", "--", *sorted(remaining)), check=True)
        remaining = set()
    if not manifest.get("needs_native"):
        status = {"status": "not_needed"}
    elif remaining:
        status = {"status": "unavailable", "reason": "resolve the merge conflicts before rebuilding"}
    else:
        env = dict(os.environ)
        env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env["PATH"]
        try:
            result = run_build(tmp, env)
            status = {"status": "ready" if result["exit_code"] == 0 and not result["timed_out"] else "failed",
                      "exit_code": result["exit_code"], "timed_out": result["timed_out"]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            log = getattr(exc, "stdout", None) or str(exc)
            (tmp / "native-build.log").write_text(log.decode(errors="replace") if isinstance(log, bytes) else log)
            status = {"status": "failed", "reason": "native build unavailable or exceeded five minutes"}
    (tmp / "native.json").write_text(json.dumps(status) + "\n")
    print("native verification:", json.dumps(status))
    return status


def main():
    tmp = Path(os.environ["RUNNER_TEMP"])
    status = build(tmp)
    context = f"\n## Native verification\n{json.dumps(status)}\n"
    if status["status"] in {"failed", "unavailable"}:
        context += f"Build diagnostics: {tmp / 'native-build.log'}. Do not claim native tests passed or silently accept skipped native tests.\n"
    context += f"After an engine/Rust fix, rebuild with this trusted helper: python {Path(__file__).resolve()}\n"
    with (tmp / "context.md").open("a") as f:
        f.write(context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
