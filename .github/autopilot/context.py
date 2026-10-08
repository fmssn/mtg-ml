"""Prepare review artifacts independently of GitHub mutations (also used by replays)."""

import argparse
import fnmatch
import json
import os
import subprocess
from pathlib import Path

EXCLUSIONS = ("data/*.json", "*.lock", "tests/data/**", "docs/experiments/ledger.jsonl")
PROMPT_BYTES = 120_000
INVARIANTS = """The Python engine is the reference; engine/rules changes must match Rust.
Preserve hidden information: a player's observations must not expose the other seat's private state.
Preserve feature layouts, checkpoint compatibility and intentional golden behavior.
Card definitions are shared in cards.toml. Do not change tests, golden digests, card data or .github/
except to resolve a listed conflict in that exact file. No cosmetic edits or redesigns.
The controller commits/pushes; lint, Python tests and native differential CI gate merging.
"""


def git(*args):
    return subprocess.check_output(("git", *args), text=True).strip()


def write_context(tmp, mode, pr, base, title, body="", ci_log="", native_ci=False):
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    head = git("rev-parse", "HEAD")
    base_sha = git("rev-parse", base)
    merge_base = git("merge-base", head, base_sha)
    paths = subprocess.check_output(("git", "diff", "--name-only", "-z", merge_base, head)).decode().split("\0")
    files = [{"path": p, "excluded": next((x for x in EXCLUSIONS if fnmatch.fnmatch(p, x)), None)} for p in paths if p]
    specs = ["."] + [f":(exclude){x}" for x in EXCLUSIONS]
    diff = subprocess.check_output(("git", "diff", merge_base, head, "--", *specs))
    (tmp / "pr.diff").write_bytes(diff)
    # Keep the completely unfiltered patch too, with exclusions explicit in the manifest.
    (tmp / "pr-full.diff").write_bytes(subprocess.check_output(("git", "diff", merge_base, head)))
    # End the embedded excerpt at a line boundary. The full patch is always available locally.
    excerpt = diff[:PROMPT_BYTES]
    if len(diff) > PROMPT_BYTES:
        excerpt = excerpt.rsplit(b"\n", 1)[0] + b"\n"
    needs_native = native_ci or any(p.startswith(("native/", "mtg_ml/", "tests/")) for p in paths)
    manifest = {"version": 2, "pr": str(pr), "mode": mode, "head": head, "base": base_sha,
                "merge_base": merge_base, "files": files, "needs_native": needs_native,
                "diff_bytes": len(diff), "embedded_diff_bytes": len(excerpt)}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (tmp / "start_head").write_text(head + "\n")
    listing = "\n".join(f"- {f['path']}" + (f" [generated artifact excluded from embedded/review diff: {f['excluded']}]" if f["excluded"] else "") for f in files)
    context = f"""# PR #{pr}: {title} (mode: {mode})

{body}

## Trusted project invariants
{INVARIANTS}
## Exact snapshot
Head: {head}
Base: {base} = {base_sha}
Merge base: {merge_base}
Use these commits, never assume origin/main or search old PR/commit history.

## Complete changed-file manifest
{listing or '(no changed files)'}

Full review patch: {tmp / 'pr.diff'} ({len(diff)} bytes).
Unfiltered patch including generated artifacts: {tmp / 'pr-full.diff'}.
Coverage manifest: {tmp / 'manifest.json'}.
Generated artifacts are intentionally excluded from code review, not silently omitted.
Read omitted review hunks from pr.diff before claiming their files reviewed.

## Review diff ({'TRUNCATED excerpt; full patch must be consulted' if len(diff) > PROMPT_BYTES else 'complete'})
```diff
{excerpt.decode(errors='replace')}
```
"""
    if ci_log:
        (tmp / "ci-failed.log").write_text(ci_log)
        ci_lines = ci_log.splitlines()
        context += f"\n## Failed CI\nFull failed log: {tmp / 'ci-failed.log'}\n" + ("TRUNCATED excerpt: last 300 lines only.\n" if len(ci_lines) > 300 else "Complete failed log:\n") + "```\n" + "\n".join(ci_lines[-300:]) + "\n```\n"
    if mode == "escalate":
        context += "\n## Specific blocker\n" + os.environ.get("ESCALATION", "Escalated explicitly by the developer.") + "\n"
    (tmp / "context.md").write_text(context)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("pr")
    p.add_argument("base")
    args = p.parse_args()
    info = json.loads(subprocess.check_output(("gh", "pr", "view", args.pr, "--json", "title,body")))
    log, native_ci = "", False
    if args.mode == "ci-fix" and os.environ.get("CI_RUN_ID"):
        run = os.environ["CI_RUN_ID"]
        log = subprocess.check_output(("gh", "run", "view", run, "--log-failed"), text=True)
        native_ci = "native\t" in log
    write_context(os.environ["RUNNER_TEMP"], args.mode, args.pr, args.base, info["title"], info.get("body") or "", log, native_ci)


if __name__ == "__main__":
    main()
