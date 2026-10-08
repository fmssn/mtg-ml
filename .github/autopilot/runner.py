"""Run a reviewer with a deadline and one evidence-preserving finalization attempt."""

import argparse
import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

from review_result import InvalidResult, completion_problems, parse_result, read_json

TOOLS = ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]
ALLOWED = ["Read", "Glob", "Grep", "Edit", "Write", "Bash(ruff check *)", "Bash(python -m pytest *)",
           "Bash(git diff *)", "Bash(git status *)", "Bash(git show *)"]
FINISH_PROMPT = """Your investigation has ended. Finalize using ONLY the existing diff, evidence,
edits and test results in this session. Tools are disabled. Do not restart the review.
List reviewed_files and pending_checks accurately. If any non-excluded changed file remains
unreviewed, a necessary check is missing, or an investigation is unfinished, verdict MUST be
incomplete (explain the remaining work). Never turn a budget limit into a clean verdict.
Use escalate only for a concrete evidenced defect, protected-file guard, or specific design
decision. Return the fenced JSON contract from your original instructions, nothing after it.
"""


def duration(value):
    suffix = value[-1:]
    factor = {"s": 1, "m": 60, "h": 3600}.get(suffix, 1)
    return float(value[:-1] if suffix in "smh" else value) * factor


def log_event(event):
    if event.get("type") == "system" and event.get("subtype") == "init":
        print(f"session: model={event.get('model')} permissionMode={event.get('permissionMode')}", flush=True)
    if event.get("type") == "assistant":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "tool_use":
                value = block.get("input", {})
                print("tool:", block.get("name"), str(value.get("command") or value.get("file_path") or value.get("pattern") or "")[:160], flush=True)


def run_phase(command, prompt, seconds, log_path, env):
    started = time.monotonic()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, start_new_session=True, env=env)
    lines = queue.Queue()

    def drain():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    process.stdin.write(prompt)
    process.stdin.close()
    result, session_id, timed_out = None, None, False
    with Path(log_path).open("w") as log:
        while True:
            remaining = seconds - (time.monotonic() - started)
            if remaining <= 0 and process.poll() is None and not timed_out:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if timed_out and process.poll() is None and remaining < -3:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                break
            log.write(line)
            log.flush()
            try:
                event = json.loads(line)
            except ValueError:
                print(line.rstrip(), flush=True)
                continue
            if not isinstance(event, dict):
                continue
            session_id = event.get("session_id") or session_id
            log_event(event)
            if event.get("type") == "result":
                result = event
    code = process.wait()
    thread.join(timeout=1)
    return {"exit_code": code, "timed_out": timed_out, "result": result,
            "session_id": session_id, "duration_seconds": round(time.monotonic() - started, 3)}


def failure_kind(phase):
    if phase["timed_out"] or (phase["result"] or {}).get("subtype") == "error_max_turns":
        return "capacity"
    if not phase["result"] or phase["exit_code"] or phase["result"].get("is_error"):
        return "infrastructure"
    return "result"


def execute(prompt_path, model, budget, timeout="15m", turns=None):
    tmp = Path(os.environ["RUNNER_TEMP"])
    tmp.mkdir(parents=True, exist_ok=True)
    deepseek = os.environ.get("AUTOPILOT_PROVIDER") == "deepseek" or "api.deepseek.com" in os.environ.get("ANTHROPIC_BASE_URL", "")
    if deepseek:
        model = os.environ.get("DEEPSEEK_MODEL") or "deepseek-flash"
    # Older workflow files may pass only four arguments.
    turns = int(turns or (60 if deepseek else 40))
    total = duration(timeout)
    reserve = min(120, total / 4) if deepseek else 0
    started = time.monotonic()
    deadline = started + total
    allowed = ALLOWED + [f"Bash(python {Path(__file__).with_name('native.py')})"]
    env = dict(os.environ)
    env.update(CLAUDE_CODE_DISABLE_BACKGROUND_TASKS="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
               BASH_DEFAULT_TIMEOUT_MS="180000", BASH_MAX_TIMEOUT_MS="300000")
    base = [os.environ.get("AUTOPILOT_CLAUDE_BIN", "claude"), "-p", "--output-format", "stream-json", "--verbose",
            "--model", model, "--permission-mode", "dontAsk", "--permission-prompts", "none"]
    if deepseek:
        env["CLAUDE_CODE_EFFORT_LEVEL"] = "max"
        # Bare is DeepSeek-only: Opus continues to use subscription OAuth.
        base += ["--bare", "--effort", "max", "--settings", '{"sandbox":{"enabled":false},"alwaysThinkingEnabled":true}']
    else:
        base += ["--max-budget-usd", budget, "--settings", '{"sandbox":{"enabled":false}}']
    base += ["--append-system-prompt", Path(prompt_path).read_text()]
    context = (tmp / "context.md").read_text()
    first = run_phase(base + ["--max-turns", str(turns), "--tools", *TOOLS, "--allowedTools", *allowed],
                      "Review and complete the PR below according to the supplied review instructions.\n\n" + context,
                      max(0.1, total - reserve), tmp / "claude.initial.jsonl", env)
    phases = [first]
    envelope = first["result"]
    invalid = None
    try:
        parse_result(envelope)
    except InvalidResult as exc:
        invalid = str(exc)
    # A malformed successful answer may be repaired from the existing session too.
    resumable = failure_kind(first) != "infrastructure"
    if deepseek and (invalid or first["timed_out"]) and resumable and first["session_id"] and deadline > time.monotonic():
        print("DeepSeek finalization: resuming the same session, tools disabled", flush=True)
        last = run_phase(base + ["--resume", first["session_id"], "--max-turns", "10", "--tools", ""],
                         FINISH_PROMPT, min(reserve, deadline - time.monotonic()), tmp / "claude.finish.jsonl", env)
        phases.append(last)
        envelope = last["result"]
    last = phases[-1]
    error, verdict = None, None
    try:
        if last["timed_out"] or last["exit_code"]:
            raise InvalidResult("model did not exit successfully within its budget")
        result = parse_result(envelope)
        pending = completion_problems(result, read_json(tmp / "manifest.json"))
        verdict = result["verdict"]
        if pending and verdict in {"clean", "fixed", "incomplete"}:
            error = "; ".join(pending)
    except InvalidResult as exc:
        error = str(exc)
    status = "complete" if not error and verdict in {"clean", "fixed"} else "blocker" if not error and verdict == "escalate" else "incomplete"
    kind = "capacity" if verdict == "incomplete" else failure_kind(last) if error else None
    # Keep the old receipt location for finish.py and older workflow callers.
    (tmp / "claude.json").write_text(json.dumps(envelope or {}) + "\n")
    # SDK costs on resume are conversation totals. Report the final total once, not a sum.
    receipt = {"status": status, "failure_kind": kind, "reason": error,
               "continued": len(phases) > 1, "model": model, "effort": "max" if deepseek else None,
               "duration_seconds": round(time.monotonic() - started, 3),
               "phases": [{k: v for k, v in p.items() if k != "result"} for p in phases],
               "conversation_cost_estimate_usd": (envelope or {}).get("total_cost_usd"),
               "usage": (envelope or {}).get("usage", {}), "models": (envelope or {}).get("modelUsage", {}),
               "denied_calls": len((envelope or {}).get("permission_denials", []))}
    (tmp / "runner.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("review receipt:", json.dumps(receipt), flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("model")
    p.add_argument("budget")
    p.add_argument("timeout", nargs="?", default="15m")
    p.add_argument("turns", nargs="?")
    args = p.parse_args()
    try:
        execute(args.prompt, args.model, args.budget, args.timeout, args.turns)
    except Exception as exc:
        # The following guard step owns the failure handoff; do not skip it.
        tmp = Path(os.environ["RUNNER_TEMP"])
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "claude.json").write_text("{}\n")
        (tmp / "runner.json").write_text(json.dumps({"status": "incomplete", "failure_kind": "infrastructure", "reason": str(exc)}) + "\n")
        print("review infrastructure failed:", type(exc).__name__, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
