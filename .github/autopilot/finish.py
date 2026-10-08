"""Guard, commit, push and decide after a model run.

Usage: finish.py <fix|escalate> <pr> <head-branch> <round> [<base-branch>]
Reads $RUNNER_TEMP/claude.json (claude --output-format json), $RUNNER_TEMP/conflicts.txt
and $RUNNER_TEMP/start_head (written by prepare.sh).
Writes `outcome` (automerge | escalate | human | retry | superseded | error) and `reason`
to $GITHUB_OUTPUT.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from review_result import InvalidResult, completion_problems, parse_result as validate_result, read_json

# The cheap model may not touch these (except to resolve a conflict in that file).
# For the escalation model, edits here are allowed but stop auto-merge.
SENSITIVE = ("tests/", "mtg_ml/engine/cards.toml")
# Nobody's edits here get pushed by the autopilot.
FORBIDDEN = (".github/",)
MARKER = re.compile(r"^(<<<<<<<|>>>>>>>) ", re.M)


def sh(*args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode:
        print(f"$ {' '.join(args)}\n{r.stdout}{r.stderr}", file=sys.stderr)
        r.check_returncode()
    return r.stdout


def gh(*args):
    # Label, comment and merge failures must not produce a successful receipt.
    subprocess.run(("gh", *args), check=True)


def output(**kv):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        for k, v in kv.items():
            f.write(f"{k}<<EOF_AUTOPILOT\n{v}\nEOF_AUTOPILOT\n")


def parse_result(tmp):
    try:
        return validate_result(read_json(tmp / "claude.json"))
    except InvalidResult:
        return None


def handoff(pr, stage, result, reason, kind="incomplete"):
    gh("pr", "merge", pr, "--disable-auto")
    gh("pr", "edit", pr, "--add-label", "needs-human")
    gh("pr", "edit", pr, "--remove-label", "needs-opus")
    comment(pr, f"{kind}; nothing pushed", result, reason)
    output(outcome="human", reason=f"{stage}: {kind}: {reason}")


def comment(pr, title, result, extra=""):
    lines = [f"**Autopilot: {title}**", "", result.get("summary", "")]
    for f in result.get("findings", []):
        lines.append(f"- `{f.get('file')}:{f.get('line')}` {f.get('issue')} ({f.get('status')})")
    if result.get("escalation_reason"):
        lines += ["", f"Escalation: {result['escalation_reason']}"]
    if extra:
        lines += ["", extra]
    gh("pr", "comment", pr, "--body", "\n".join(lines))


def push(head, base):
    """Push HEAD to the PR branch. Returns None on success, else the outcome to report.

    If the branch moved during the run only because the base branch was merged in
    (update-branch after something landed on the default branch), merge that in and
    push again instead of throwing the whole review away.
    """
    try:
        sh("git", "push", "origin", f"HEAD:refs/heads/{head}")
        return None
    except subprocess.CalledProcessError:
        pass
    sh("git", "fetch", "-q", "origin", head, base)
    if not subprocess.run(("git", "merge-base", "--is-ancestor", f"origin/{head}", "HEAD")).returncode:
        raise RuntimeError("push failed although the branch did not move")
    start = (Path(os.environ["RUNNER_TEMP"]) / "start_head").read_text().strip()
    # Non-merge commits on the remote branch that are neither ours nor from the base.
    foreign = sh("git", "rev-list", "--no-merges", f"origin/{head}", f"^{start}", f"^origin/{base}").strip()
    if foreign:
        return "superseded"   # the developer pushed; that push triggered its own review
    if subprocess.run(("git", "merge", "--no-edit", f"origin/{head}"), capture_output=True).returncode:
        sh("git", "merge", "--abort", check=False)
        return "retry"
    try:
        sh("git", "push", "origin", f"HEAD:refs/heads/{head}")
        return None
    except subprocess.CalledProcessError:
        return "retry"


def main():
    stage, pr, head, rnd = sys.argv[1:5]
    base = sys.argv[5] if len(sys.argv) > 5 else os.environ["BASE_REF"]
    tmp = Path(os.environ["RUNNER_TEMP"])
    conflicts = set((tmp / "conflicts.txt").read_text().split())
    if stage == "fix":
        gh("pr", "edit", pr, "--add-label", f"autopilot:round-{rnd}")

    result = parse_result(tmp)
    receipt = read_json(tmp / "runner.json", {})

    # Files the model touched: unstaged edits and new files. The merge's own
    # changes from the base branch are already staged, so they don't show here.
    touched = set(filter(None, sh("git", "diff", "--name-only", "-z").split("\0")))
    touched |= set(filter(None, sh("git", "ls-files", "--others", "--exclude-standard", "-z").split("\0")))
    own = touched - conflicts
    forbidden = sorted(p for p in own if p.startswith(FORBIDDEN))
    sensitive = sorted(p for p in own if p.startswith(SENSITIVE))
    markers = sorted(p for p in touched | conflicts if Path(p).is_file() and MARKER.search(Path(p).read_text(errors="ignore")))
    lint = subprocess.run(("ruff", "check", "."), capture_output=True, text=True)

    problems = []
    if forbidden:
        problems.append(f"edited files the autopilot may not push: {', '.join(forbidden)}")
    if stage == "fix" and sensitive:
        problems.append(f"edited tests or card data: {', '.join(sensitive)}")
    if markers:
        problems.append(f"conflict markers left in: {', '.join(markers)}")
    fallback = {"summary": "Reviewer did not produce a complete valid verdict.", "findings": []}
    if not problems:
        if result is None or receipt.get("status") == "incomplete":
            envelope = read_json(tmp / "claude.json", {})
            kind = receipt.get("failure_kind") or ("capacity" if envelope.get("subtype") == "error_max_turns" else "result")
            reason = receipt.get("reason") or "Model produced no usable verdict; capacity or API failure is not an Opus escalation."
            handoff(pr, stage, result or fallback, reason, kind)
            return 0
        pending = completion_problems(result, read_json(tmp / "manifest.json"))
        if result["verdict"] in ("clean", "fixed", "incomplete") and pending:
            handoff(pr, stage, result, "; ".join(pending))
            return 0
        if lint.returncode:
            problems.append("ruff check fails after the fix")
        native = read_json(tmp / "native.json", {})
        if native.get("status") in ("failed", "unavailable") and result["verdict"] in ("clean", "fixed"):
            handoff(pr, stage, result, "Native verification is unavailable; review cannot claim completion.", "verification")
            return 0
        if result["verdict"] == "escalate":
            problems.append(result.get("escalation_reason") or "unresolved evidenced defect")

    if problems:
        result = result or fallback
        reason = "; ".join(problems)
        gh("pr", "merge", pr, "--disable-auto")
        if stage == "fix":
            gh("pr", "edit", pr, "--add-label", "needs-opus")
            comment(pr, "escalating to Opus, nothing pushed", result, reason)
            output(outcome="escalate", reason=f"{reason}\n\nCheaper model's report:\n{json.dumps(result, indent=1)}")
        else:
            gh("pr", "edit", pr, "--add-label", "needs-human")
            gh("pr", "edit", pr, "--remove-label", "needs-opus")
            comment(pr, "needs a human decision, nothing pushed", result, reason)
            output(outcome="human", reason=reason)
        return 0

    sh("git", "add", "-A")
    in_merge = Path(sh("git", "rev-parse", "--git-path", "MERGE_HEAD").strip()).exists()
    if in_merge or sh("git", "diff", "--cached", "--name-only").strip():
        msg = f"autopilot ({stage}): {result.get('summary', 'fixes')}"[:200]
        sh("git", "commit", "--no-verify", "-m", msg)
    if sh("git", "rev-list", f"origin/{head}..HEAD").strip():
        moved = push(head, base)
        if moved:
            # Drop this round. A developer push starts its own review run; anything
            # else (a base update we couldn't merge cleanly) is re-dispatched.
            if stage == "fix":
                gh("pr", "edit", pr, "--remove-label", f"autopilot:round-{rnd}")
            else:
                gh("pr", "edit", pr, "--remove-label", "autopilot:opus-used")
            output(outcome=moved, reason=f"{head} moved during the run")
            print(f"{head} moved during the run; nothing pushed ({moved})", file=sys.stderr)
            return 0

    if result.get("findings") or result.get("verdict") == "fixed":
        comment(pr, "fixed and pushed", result)
    if stage == "escalate" and sensitive:
        gh("pr", "edit", pr, "--add-label", "needs-human")
        gh("pr", "edit", pr, "--remove-label", "needs-opus")
        gh("pr", "merge", pr, "--disable-auto")
        gh("pr", "comment", pr, "--body", f"**Autopilot:** Opus changed tests or card data ({', '.join(sensitive)}). Auto-merge is off; please check the change.")
        output(outcome="human", reason="escalation model changed tests or card data")
        return 0
    gh("pr", "edit", pr, "--remove-label", "needs-opus")
    gh("pr", "merge", pr, "--auto", "--squash", "--match-head-commit", sh("git", "rev-parse", "HEAD").strip())
    output(outcome="automerge", reason=result.get("summary", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
