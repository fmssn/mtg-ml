"""Bounded native build: failure is actionable review context, not a skipped model run."""

import json
import os
import subprocess
import sys
from pathlib import Path

from review_result import read_json


def build(tmp):
    tmp = Path(tmp)
    manifest = read_json(tmp / "manifest.json", {})
    if not manifest.get("needs_native"):
        status = {"status": "not_needed"}
    elif subprocess.run(("git", "diff", "--name-only", "--diff-filter=U"), capture_output=True, text=True).stdout.strip():
        status = {"status": "unavailable", "reason": "resolve the merge conflicts before rebuilding"}
    else:
        env = dict(os.environ)
        env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env["PATH"]
        try:
            result = subprocess.run(("maturin", "develop", "--release", "--locked", "--manifest-path", "native/Cargo.toml"),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300, env=env)
            (tmp / "native-build.log").write_text(result.stdout)
            status = {"status": "ready" if result.returncode == 0 else "failed", "exit_code": result.returncode}
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
