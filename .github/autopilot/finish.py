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
    subprocess.run(("gh", *args), check=False)


def output(**kv):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        for k, v in kv.items():
            f.write(f"{k}<<EOF_AUTOPILOT\n{v}\nEOF_AUTOPILOT\n")


def parse_result(tmp):
    try:
        text = json.loads((tmp / "claude.json").read_text()).get("result", "")
        blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S)
        return json.loads(blocks[-1])
    except (OSError, ValueError, IndexError):
        return None


def ran_out_of_turns(tmp):
    try:
        return json.loads((tmp / "claude.json").read_text()).get("subtype") == "error_max_turns"
    except (OSError, ValueError):
        return False


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
    if result is None and stage == "fix" and ran_out_of_turns(tmp):
        result = {"verdict": "escalate", "summary": "The cheap model hit its turn cap without a verdict.",
                  "findings": [], "escalation_reason": "cheap model hit its turn cap"}
    if result is None:
        output(outcome="error", reason="model run produced no parsable result")
        print("no parsable result; nothing pushed", file=sys.stderr)
        if stage == "escalate":
            # e.g. an expired CLAUDE_CODE_OAUTH_TOKEN (401): don't leave the PR in limbo.
            gh("pr", "merge", pr, "--disable-auto")
            gh("pr", "edit", pr, "--add-label", "needs-human")
            gh("pr", "edit", pr, "--remove-label", "needs-opus")
            run = f"{os.environ.get('GITHUB_SERVER_URL')}/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"
            gh("pr", "comment", pr, "--body", f"**Autopilot:** the Opus run failed without a result (auth or API error?), nothing pushed. Over to you. Log: {run}")
        return 1

    # Files the model touched: unstaged edits and new files. The merge's own
    # changes from the base branch are already staged, so they don't show here.
    touched = set(sh("git", "diff", "--name-only").split())
    touched |= set(sh("git", "ls-files", "--others", "--exclude-standard").split())
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
    if lint.returncode:
        problems.append("ruff check fails after the fix")
    if result.get("verdict") not in ("clean", "fixed"):
        problems.append(result.get("escalation_reason") or "model asked to escalate")

    if problems:
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
    gh("pr", "merge", pr, "--auto", "--squash")
    output(outcome="automerge", reason=result.get("summary", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
