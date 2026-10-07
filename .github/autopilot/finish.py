"""Guard, commit, push and decide after a model run.

Usage: finish.py <fix|escalate> <pr> <head-branch> <round>
Reads $RUNNER_TEMP/claude.json (claude --output-format json) and $RUNNER_TEMP/conflicts.txt.
Writes `outcome` (automerge | escalate | human | retry | error) and `reason` to $GITHUB_OUTPUT.
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


def comment(pr, title, result, extra=""):
    lines = [f"**Autopilot: {title}**", "", result.get("summary", "")]
    for f in result.get("findings", []):
        lines.append(f"- `{f.get('file')}:{f.get('line')}` {f.get('issue')} ({f.get('status')})")
    if result.get("escalation_reason"):
        lines += ["", f"Escalation: {result['escalation_reason']}"]
    if extra:
        lines += ["", extra]
    gh("pr", "comment", pr, "--body", "\n".join(lines))


def main():
    stage, pr, head, rnd = sys.argv[1:5]
    tmp = Path(os.environ["RUNNER_TEMP"])
    conflicts = set((tmp / "conflicts.txt").read_text().split())
    if stage == "fix":
        gh("pr", "edit", pr, "--add-label", f"autopilot:round-{rnd}")

    result = parse_result(tmp)
    if result is None:
        output(outcome="error", reason="model run produced no parsable result")
        print("no parsable result; nothing pushed", file=sys.stderr)
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
        try:
            sh("git", "push", "origin", f"HEAD:refs/heads/{head}")
        except subprocess.CalledProcessError:
            sh("git", "fetch", "-q", "origin", head)
            if not subprocess.run(("git", "merge-base", "--is-ancestor", f"origin/{head}", "HEAD")).returncode:
                raise
            # The developer pushed while we worked. Drop this round and run again on the new head.
            if stage == "fix":
                gh("pr", "edit", pr, "--remove-label", f"autopilot:round-{rnd}")
            else:
                gh("pr", "edit", pr, "--remove-label", "autopilot:opus-used")
            output(outcome="retry", reason=f"{head} moved during the run")
            print(f"{head} moved during the run; nothing pushed, retrying", file=sys.stderr)
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
