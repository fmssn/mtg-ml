"""Replay the audited reviews with no GitHub writes, commits, pushes or auto-merges.

Each case gets its own checkout, environment, native extension and receipt directory.
Provider keys stay in subprocess environments. Artifacts are resumable and gitignored.
"""

import argparse
import concurrent.futures
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AP = ROOT / ".github/autopilot"
AUDITED = {31: 37672572444, 32: 37672566641, 33: 37677228102, 34: 37677229060,
           35: 37683075193, 36: 37680980016, 41: 37672700893, 43: 37688128777,
           45: 37737139995, 47: 37735939773, 48: 37739763783, 50: 37768035382, 51: 37768115353}
CONTROL_FIXES = {"hidden-card": "e319b0973d10e55d77d3bf30749836c2c2871649",
                 "native-shutdown": "176936930d9ba8027b8df3311690d187981619aa"}
REPLAY_VERSION = 2  # v2 merges the historical base before native build/review.


def command(*args, cwd=ROOT, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def gh(path):
    return json.loads(command("gh", "api", path))


def available_commit(sha):
    if subprocess.run(("git", "cat-file", "-e", sha + "^{commit}"), cwd=ROOT, capture_output=True).returncode:
        command("git", "fetch", "-q", "origin", sha)
    return command("git", "rev-parse", sha)


def checkout_snapshot(log):
    """Dispatch run head_sha identifies the workflow, not the checked-out PR."""
    lines = log.splitlines()
    shas = []
    for i, line in enumerate(lines[:-1]):
        if "[command]" in line and "git log -1 --format=%H" in line:
            match = re.search(r"\b([a-f0-9]{40})$", lines[i + 1])
            if match:
                shas.append(match[1])
    if len(shas) < 2:
        raise ValueError("Replay requires both PR and trusted-base checkout SHAs from the original run log")
    return shas[0], shas[1]


def discover():
    cases = []
    for pr, run_id in AUDITED.items():
        info = gh(f"repos/fmssn/mtg-ml/pulls/{pr}")
        head, base = checkout_snapshot(command("gh", "run", "view", str(run_id), "--log"))
        cases.append({"id": f"pr-{pr}", "pr": pr, "run_id": run_id, "kind": "historical",
                      "head": available_commit(head), "base": available_commit(base), "snapshot_source": "original checkout log",
                      "title": info["title"], "body": info.get("body") or ""})
    info = gh("repos/fmssn/mtg-ml/pulls/30")
    for bug, fix in CONTROL_FIXES.items():
        fix = available_commit(fix)
        before = command("git", "rev-parse", fix + "^1")
        # Use the branch's original fork; do not hint which bug the model should find.
        base = command("git", "merge-base", fix, "1ee10e2a6599d53fdccbb8ddda5b9aec6735b2b3^1")
        for label, head in (("before", before), ("after", fix)):
            cases.append({"id": f"pr-30-{bug}-{label}", "pr": 30, "kind": "control", "bug": bug,
                          "expect_finding": label == "before", "head": head, "base": base,
                          "title": info["title"], "body": info.get("body") or ""})
    return cases


def provider_env(output):
    env = dict(os.environ)
    key = env.get("DEEPSEEK_API_KEY")
    if not key:
        dotenv = ROOT / ".env"
        if dotenv.exists():
            for line in dotenv.read_text().splitlines():
                name, sep, value = line.partition("=")
                if sep and name.strip() == "DEEPSEEK_API_KEY":
                    key = value.strip().strip("\"'")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for real shadow replays")
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
        env.pop(name, None)
    env.update(ANTHROPIC_API_KEY=key, ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic",
               AUTOPILOT_PROVIDER="deepseek", DEEPSEEK_MODEL="deepseek-flash",
               CLAUDE_CONFIG_DIR=str(output / "claude-config"),
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env["PATH"]
    env["CARGO_TARGET_DIR"] = str(output / "cargo-target")
    return env


def matches_bug(finding, bug):
    issue = finding.get("issue", "").lower()
    if bug == "hidden-card":
        return finding.get("file") == "mtg_ml/live.py" and any(w in issue for w in ("prompt", "delver", "top card", "hidden", "library"))
    return finding.get("file") in ("mtg_ml/replay.py", "mtg_ml/live.py") and any(w in issue for w in ("shutdown", "close", "thread", "unsendable", "ctrl-c"))


def assess_checkout(record, checkout, tmp, env):
    """Apply the local guard checks without calling the GitHub controller."""
    conflicts = set((tmp / "conflicts.txt").read_text().splitlines())
    touched = set(filter(None, command("git", "diff", "--name-only", "-z", cwd=checkout).split("\0")))
    touched |= set(filter(None, command("git", "ls-files", "--others", "--exclude-standard", "-z", cwd=checkout).split("\0")))
    violations = sorted(p for p in touched - conflicts if p.startswith(("tests/", "mtg_ml/engine/cards.toml", ".github/")))
    lint = subprocess.run((str(checkout / ".venv/bin/ruff"), "check", "."), cwd=checkout, env=env,
                          capture_output=True, text=True)
    record.update(guard_violations=violations, lint_passed=lint.returncode == 0)
    record["completed"] = record["completed"] and not violations and lint.returncode == 0
    if record["kind"] == "control":
        found = any(matches_bug(f, record["bug"]) for f in (record.get("report") or {}).get("findings", []))
        record["control_passed"] = found if record["expect_finding"] else record["completed"] and not found
    return record


def replay(case, output, env):
    case_dir = output / case["id"]
    receipt_path = case_dir / "replay.json"
    if receipt_path.exists():
        cached = json.loads(receipt_path.read_text())
        exact = cached.get("head") == case["head"] and cached.get("base") == case["base"]
        # A v1 replay is faithful only if its base was already in the head.
        contained = not subprocess.run(("git", "merge-base", "--is-ancestor", case["base"], case["head"]),
                                       cwd=ROOT, capture_output=True).returncode
        if exact and (cached.get("replay_version") == REPLAY_VERSION or cached.get("replay_version", 1) == 1 and contained):
            cached["replay_version"] = REPLAY_VERSION
            assess_checkout(cached, case_dir / "checkout", case_dir / "artifacts", env)
            receipt_path.write_text(json.dumps(cached, indent=2) + "\n")
            return cached
        archive = output / "superseded"
        archive.mkdir(exist_ok=True)
        case_dir.rename(archive / f"{case['id']}-{time.time_ns()}")
    case_dir.mkdir(parents=True, exist_ok=True)
    checkout, tmp = case_dir / "checkout", case_dir / "artifacts"
    tmp.mkdir(exist_ok=True)
    started = time.monotonic()
    try:
        if not checkout.exists():
            command("git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(checkout))
            command("git", "checkout", "--quiet", "--detach", case["head"], cwd=checkout)
            # A replay clone has no remote write endpoint at all.
            command("git", "remote", "remove", "origin", cwd=checkout)
        py = checkout / ".venv/bin/python"
        env_case = dict(env, RUNNER_TEMP=str(tmp), VIRTUAL_ENV=str(checkout / ".venv"),
                        CLAUDE_CONFIG_DIR=str(case_dir / "claude-config"))
        env_case["PATH"] = str(checkout / ".venv/bin") + os.pathsep + env_case["PATH"]
        with (case_dir / "setup.log").open("w") as log:
            if not py.exists():
                subprocess.run(("uv", "venv", str(checkout / ".venv"), "--python", str(ROOT / ".venv/bin/python")), check=True, stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(("uv", "pip", "install", "--python", str(py), "-e", ".[dev]", "numpy", "torch", "ruff", "maturin>=1.5,<2"),
                           cwd=checkout, check=True, stdout=log, stderr=subprocess.STDOUT)
        sys.path.insert(0, str(AP))
        context = importlib.import_module("context")
        results = importlib.import_module("review_result")
        # The helper invokes read-only git in this case's checkout.
        # Avoid process-wide chdir: cases can run concurrently.
        prepare_code = """import sys, subprocess
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from context import write_context
m = write_context(*sys.argv[2:])
tmp = Path(sys.argv[2])
p = subprocess.run(['git','merge','--no-edit','--no-commit',m['base']], capture_output=True, text=True)
conflicts = subprocess.check_output(['git','diff','--name-only','--diff-filter=U'], text=True)
(tmp/'conflicts.txt').write_text(conflicts)
(tmp/'merge.log').write_text(p.stdout+p.stderr)
if p.returncode and not conflicts:
    raise RuntimeError('Snapshot merge failed; see merge.log')
if conflicts:
    with (tmp/'context.md').open('a') as f:
        f.write('\\n## Actual merge conflicts (permission exceptions only for these files)\\n'+conflicts)
"""
        manifest_cmd = [str(py), "-c", prepare_code,
                        str(AP), str(tmp), "review", str(case["pr"]), case["base"], case["title"], case["body"]]
        subprocess.run(manifest_cmd, cwd=checkout, env=env_case, check=True)
        assert context.INVARIANTS  # Import failures must happen before any provider call.
        with (case_dir / "review.log").open("w") as log:
            subprocess.run((str(py), str(AP / "native.py")), cwd=checkout, env=env_case, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run((str(py), str(AP / "runner.py"), str(AP / "fix.md"), "sonnet", "3", "15m", "60"),
                           cwd=checkout, env=env_case, stdout=log, stderr=subprocess.STDOUT, timeout=920, check=True)
        runner = results.read_json(tmp / "runner.json", {})
        try:
            report = results.parse_result(results.read_json(tmp / "claude.json"))
            pending = results.completion_problems(report, results.read_json(tmp / "manifest.json"))
            completed = runner.get("status") == "complete" and not pending and results.read_json(tmp / "native.json", {}).get("status") in ("ready", "not_needed")
        except results.InvalidResult as exc:
            report, pending, completed = None, [str(exc)], False
        edits = command("git", "diff", "--stat", cwd=checkout)
        (case_dir / "edits.diff").write_text(command("git", "diff", cwd=checkout) + "\n")
        record = {"id": case["id"], "kind": case["kind"], "replay_version": REPLAY_VERSION,
                  "head": case["head"], "base": case["base"],
                  "completed": completed, "report": report, "pending": pending, "runner": runner,
                  "native": results.read_json(tmp / "native.json"), "edits": edits,
                  "total_seconds": round(time.monotonic() - started, 2)}
        if case["kind"] == "control":
            found = any(matches_bug(f, case["bug"]) for f in (report or {}).get("findings", []))
            record.update(bug=case["bug"], expect_finding=case["expect_finding"], control_passed=found == case["expect_finding"])
        assess_checkout(record, checkout, tmp, env_case)
    except Exception as exc:
        # Keep credentials out of persisted errors (exceptions may carry commands/env).
        record = {"id": case["id"], "kind": case["kind"], "completed": False,
                  "error": type(exc).__name__, "total_seconds": round(time.monotonic() - started, 2)}
    receipt_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def summary(records):
    historical = [r for r in records if r["kind"] == "historical"]
    controls = [r for r in records if r["kind"] == "control"]
    runtime = sorted(r["runner"]["duration_seconds"] for r in records if r.get("runner"))
    return {"historical_complete": sum(r["completed"] for r in historical), "historical_total": len(historical),
            "controls_passed": sum(r.get("control_passed", False) for r in controls), "controls_total": len(controls),
            "median_review_seconds": runtime[len(runtime) // 2] if runtime else None,
            "acceptance_passed": len(historical) == 13 and sum(r["completed"] for r in historical) >= 11
                                 and len(controls) == 4 and all(r.get("control_passed") for r in controls),
            "machine": platform.platform(), "records": records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / ".context/autopilot-replay")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--discover-only", action="store_true")
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "cases.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(discover(), indent=2) + "\n")
    cases = json.loads(manifest_path.read_text())
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
    if args.discover_only:
        print(f"Recorded {len(cases)} exact replay cases in {manifest_path}")
        return 0
    env = provider_env(output)
    # Select the pinned cached CLI without changing the user's installed CLI.
    candidates = list((Path.home() / ".npm/_npx").glob("*/node_modules/@anthropic-ai/claude-code/package.json"))
    for candidate in candidates:
        package = json.loads(candidate.read_text())
        if package.get("version") == "2.1.293":
            env["AUTOPILOT_CLAUDE_BIN"] = str(candidate.parents[2] / ".bin/claude")
            break
    if "AUTOPILOT_CLAUDE_BIN" not in env:
        raise RuntimeError("Cache pinned Claude Code first: npm exec --yes --package @anthropic-ai/claude-code@2.1.293 -- claude --version")
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(replay, case, output, env): case for case in cases}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"case": record["id"], "completed": record["completed"], "control_passed": record.get("control_passed"), "error": record.get("error")}), flush=True)
            (output / "summary.json").write_text(json.dumps(summary(records), indent=2) + "\n")
    result = summary(records)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    return 0 if result["acceptance_passed"] or args.case else 1


if __name__ == "__main__":
    raise SystemExit(main())
