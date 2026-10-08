#!/usr/bin/env python3
"""Activate an approved plan and verify its starting state in a fresh agent chat."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from progress import write_paths


def git(root, *args):
    return subprocess.check_output(("git", "-C", str(root), *args), stderr=subprocess.PIPE)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def relative(root, path):
    return str(path.resolve().relative_to(root.resolve()))


def working_changes(root):
    """Fingerprint changes without storing source, secrets, images or full logs."""
    state = hashlib.sha256(git(root, "diff", "HEAD", "--binary", "--no-ext-diff"))
    state.update(git(root, "diff", "--cached", "--binary", "--no-ext-diff"))
    for name in sorted(git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")):
        if name:
            path = root / os.fsdecode(name)
            state.update(name + b"\0")
            if path.is_symlink():
                state.update(os.fsencode(os.readlink(path)))
            else:
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        state.update(block)
    return state.hexdigest()


def dependency(repo, number):
    return json.loads(subprocess.check_output(("gh", "pr", "view", str(number), "--repo", repo, "--json",
                                              "number,url,state,headRefOid,baseRefName"), text=True, timeout=30))


def activate(root, args):
    plan = (root / args.plan).resolve()
    plan_path = relative(root, plan)
    if not plan.is_file():
        raise ValueError(f"Plan does not exist: {plan}")
    if args.depends_on and not args.repo:
        raise ValueError("Dependency PRs require --repo owner/name.")
    branch = git(root, "symbolic-ref", "--short", "HEAD").decode().strip()
    record = {
        "version": 1,
        "task_id": args.task,
        "owner": args.owner,
        "workspace": str(root),
        "plan": plan_path,
        "plan_sha256": digest(plan.read_bytes()),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "branch": branch,
        "head_sha": git(root, "rev-parse", "HEAD").decode().strip(),
        "base_ref": args.base,
        "base_sha": git(root, "rev-parse", "--verify", args.base + "^{commit}").decode().strip(),
        "working_changes_sha256": working_changes(root),
        "repo": args.repo,
        "dependencies": [dependency(args.repo, number) for number in args.depends_on or []],
        "write_paths": write_paths(args.write_path),
        "acceptance_checks": args.check,
    }
    target = root / ".context/handoff.json"
    target.parent.mkdir(exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return record


def verify(root, record):
    errors = []
    if record.get("version") != 1:
        return ["Unsupported handoff version."]
    if record.get("workspace") != str(root):
        errors.append("Workspace changed; this handoff belongs to another checkout.")
    plan = (root / record["plan"]).resolve()
    relative(root, plan)
    if not plan.is_file() or digest(plan.read_bytes()) != record["plan_sha256"]:
        errors.append("Approved plan is missing or has changed.")
    for label, actual, expected in (
        ("Branch", git(root, "symbolic-ref", "--short", "HEAD").decode().strip(), record["branch"]),
        ("HEAD", git(root, "rev-parse", "HEAD").decode().strip(), record["head_sha"]),
        ("Base ref", git(root, "rev-parse", "--verify", record["base_ref"] + "^{commit}").decode().strip(), record["base_sha"]),
        ("Working changes", working_changes(root), record["working_changes_sha256"]),
    ):
        if actual != expected:
            errors.append(f"{label} changed since the handoff.")
    for old in record["dependencies"]:
        current = dependency(record["repo"], old["number"])
        if current != old:
            errors.append(f"Dependency PR #{old['number']} changed; recheck its state and interface.")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    activate_parser = commands.add_parser("activate", help="Record a user-approved plan; does not grant approval.")
    for field in ("plan", "task", "owner", "base"):
        activate_parser.add_argument("--" + field, required=True)
    activate_parser.add_argument("--repo")
    activate_parser.add_argument("--depends-on", type=int, action="append")
    activate_parser.add_argument("--write-path", action="append", required=True)
    activate_parser.add_argument("--check", action="append", required=True)
    commands.add_parser("check")
    commands.add_parser("show")
    args = parser.parse_args(argv)
    try:
        root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip()).resolve()
        if args.cmd == "activate":
            record = activate(root, args)
            print(f"Active plan: {record['plan']}\nHandoff: {root / '.context/handoff.json'}")
            return 0
        record = json.loads((root / ".context/handoff.json").read_text())
        if args.cmd == "show":
            print(json.dumps(record, indent=2))
            return 0
        errors = verify(root, record)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            print("Inspect the changes and reconcile the plan before activating a fresh handoff; do not refresh blindly.", file=sys.stderr)
            return 1
        print(f"Handoff matches. Read {record['plan']} before editing; checks listed there remain to be run.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"Cannot verify handoff: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
