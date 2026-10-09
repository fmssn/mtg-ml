"""Read-only merge eligibility checks, loaded from the trusted default branch."""

import argparse
import json
import subprocess
import sys
from urllib.parse import quote

REQUIRED_CHECKS = {"lint", "python", "native"}


class PolicyError(RuntimeError):
    pass


def read_json(*args):
    try:
        return json.loads(subprocess.check_output(("gh", *args), text=True, timeout=30))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise PolicyError(f"Cannot verify merge policy: {exc}") from exc


def require_target(pr, *, repo=None, expected_base=None, expected_head=None):
    """Fail closed for stacks, stale reviews, missing protection or API failures."""
    info = read_json("repo", "view", *([repo] if repo else []), "--json", "nameWithOwner,defaultBranchRef")
    repo = info["nameWithOwner"]
    default = (info.get("defaultBranchRef") or {}).get("name")
    current = read_json("pr", "view", str(pr), "--repo", repo, "--json",
                        "state,isDraft,baseRefName,headRefOid")
    if current.get("state") != "OPEN" or current.get("isDraft") is not False:
        raise PolicyError("PR must be open and ready; drafts remain with their workspace owner.")
    base = current.get("baseRefName")
    if not default or base != default:
        raise PolicyError("Stacked PR: keep it draft until its prerequisite lands, then retarget to the default branch.")
    if expected_base is not None and base != expected_base:
        raise PolicyError("PR base changed during review; validate and review the new target first.")
    if expected_head is not None and current.get("headRefOid") != expected_head:
        raise PolicyError("PR head changed during review; the reviewed commit is no longer current.")
    ref = quote(base, safe="")
    branch = read_json("api", f"repos/{repo}/branches/{ref}")
    rules = read_json("api", f"repos/{repo}/rules/branches/{ref}")
    checks = set()
    has_pr_rule = False
    for rule in rules:
        has_pr_rule |= rule.get("type") == "pull_request"
        parameters = rule.get("parameters", {})
        if rule.get("type") == "required_status_checks":
            checks.update(check["context"] for check in parameters.get("required_status_checks", []))
    if branch.get("protected") is not True or not has_pr_rule or not REQUIRED_CHECKS <= checks:
        raise PolicyError("Default branch must enforce PRs and required lint, python and native checks.")
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr")
    parser.add_argument("--repo")
    args = parser.parse_args()
    try:
        require_target(args.pr, repo=args.repo)
    except (PolicyError, KeyError, TypeError) as exc:
        print(f"Autopilot held: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
