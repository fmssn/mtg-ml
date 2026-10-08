#!/usr/bin/env python3
"""Shared task ownership, GitHub observations and cooperative resource reservations.

Install this standalone file beside the existing progress.json, or pass --file.
The legacy track/result/log commands and the existing progress.json.lock remain
compatible. This tool never creates a second tracker or changes a GitHub PR.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import subprocess
import sys
import tempfile

ACTIVE = {"running", "review", "blocked"}
STATUSES = ("todo", "running", "review", "done", "blocked", "failed")


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write(path, data):
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            os.fchmod(stream.fileno(), path.stat().st_mode & 0o777)
            json.dump(data, stream, indent=1)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def transaction(path):
    if not path.is_file():
        raise ValueError(f"Existing tracker required: {path}; use --file or MTG_PROGRESS_PATH.")
    with Path(str(path) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text())
        yield data
        data["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data["log"] = data.get("log", [])[-50:]
        atomic_write(path, data)


def event(data, message):
    data.setdefault("log", []).append({"t": datetime.now().strftime("%H:%M"), "msg": message})


def task(data, task_id):
    found = next((item for item in data["tracks"] if item["id"] == task_id), None)
    if found is None:
        raise ValueError(f"Unknown task: {task_id}")
    return found


def check_owner(item, owner, *, required=False):
    if (required and not item.get("owner")) or (item.get("owner") and item["owner"] != owner):
        raise ValueError(f"Task {item['id']} is owned by {item.get('owner') or 'nobody'}; supply its --owner.")


def write_paths(values):
    result = []
    for value in values:
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or str(path) == "." or re.search(r"[?*\[\\]", value):
            raise ValueError(f"Use literal repository-relative files/directories for ownership: {value}")
        result.append(str(path))
    return sorted(set(result))


def overlaps(left, right):
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def check_claims(data, candidate):
    if candidate.get("status") not in ACTIVE:
        return
    for other in data["tracks"]:
        if other["id"] == candidate["id"] or other.get("status") not in ACTIVE:
            continue
        for left in candidate.get("write_paths", []):
            for right in other.get("write_paths", []):
                if overlaps(left, right):
                    raise ValueError(f"Write ownership conflict: {left} overlaps {other['id']} ({other.get('owner')}): {right}")


def update_track(data, args):
    item = next((item for item in data["tracks"] if item["id"] == args.id), None)
    if item is None:
        if not args.name:
            raise ValueError(f"Unknown task {args.id}; use --name to add it.")
        item = {"id": args.id, "name": args.name, "status": "todo", "steps": []}
        data["tracks"].append(item)
    check_owner(item, args.owner)
    for field in ("name", "status", "note", "pr", "branch", "owner", "workspace", "handoff"):
        value = getattr(args, field)
        if value is not None:
            if not value.strip():
                raise ValueError(f"{field} must not be empty")
            item[field] = value
    if args.write_path is not None:
        if not item.get("owner"):
            raise ValueError("Write claims require --owner, --workspace and --branch.")
        item["write_paths"] = write_paths(args.write_path)
    if item.get("owner") and not all(item.get(key) for key in ("workspace", "branch", "write_paths")):
        raise ValueError("Owned tasks require a workspace, branch and explicit write paths.")
    if args.depends_on is not None:
        item["dependencies"] = args.depends_on
    for step in args.step or []:
        name, _, flag = step.partition("=")
        existing = next((entry for entry in item["steps"] if entry["name"] == name), None)
        if existing is None:
            item["steps"].append({"name": name, "done": flag == "done"})
        else:
            existing["done"] = flag == "done"
    check_claims(data, item)
    if item.get("status") not in ACTIVE and any(r["task_id"] == item["id"] for r in data.get("resources", [])):
        raise ValueError("Release the task's resource reservations before finishing it.")
    item["updated_at"] = timestamp()
    if args.note is not None:
        item["note_updated_at"] = item["updated_at"]
    if args.status:
        event(data, f"{item['name']}: {args.status}" + (f" ({args.note})" if args.note else ""))


def github(*args):
    return json.loads(subprocess.check_output(("gh", *args), text=True, timeout=30))


def reconcile(path, repo):
    """Fetch without holding the writer lock; apply only to unchanged PR bindings."""
    before = json.loads(path.read_text())
    observations, errors = [], []
    fields = "number,url,state,isDraft,headRefOid,baseRefName,mergedAt"
    for item in before["tracks"]:
        pr, branch = item.get("pr"), item.get("branch")
        if not pr and not branch:
            continue
        try:
            if pr:
                if not re.fullmatch(r"\d+|https://github\.com/" + re.escape(repo) + r"/pull/\d+", str(pr)):
                    raise ValueError(f"PR reference does not belong to {repo}: {pr}")
                observed = github("pr", "view", str(pr), "--repo", repo, "--json", fields)
            else:
                matches = github("pr", "list", "--repo", repo, "--state", "all", "--head", branch,
                                 "--limit", "1", "--json", fields)
                if not matches:
                    continue
                observed = matches[0]
            observed["observed_at"] = timestamp()
            observations.append((item["id"], pr, branch, observed))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append((item["id"], pr, branch, str(exc)))
    with transaction(path) as data:
        for task_id, old_pr, old_branch, observed in observations:
            item = task(data, task_id)
            if (item.get("pr"), item.get("branch")) == (old_pr, old_branch):
                item["github"] = observed
                item["pr"] = observed["url"]
                item.pop("github_error", None)
        for task_id, old_pr, old_branch, message in errors:
            item = task(data, task_id)
            if (item.get("pr"), item.get("branch")) == (old_pr, old_branch):
                item["github_error"] = {"message": message, "observed_at": timestamp()}
        event(data, f"GitHub reconciliation: {len(observations)} observations, {len(errors)} errors; task completion unchanged.")
    for task_id, _, _, message in errors:
        print(f"{task_id}: {message}", file=sys.stderr)
    return bool(errors)


def reserve(path, task_id, owner, names):
    if not names or any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Supply distinct, nonempty resource names.")
    with transaction(path) as data:
        item = task(data, task_id)
        check_owner(item, owner, required=True)
        if item["status"] not in ACTIVE:
            raise ValueError("Only active tasks can reserve resources.")
        resources = data.setdefault("resources", [])
        for reservation in resources:
            if reservation["name"] in names:
                raise ValueError(f"Resource busy: {reservation['name']} held by {reservation['task_id']} ({reservation['owner']})")
        resources.extend({"name": name, "task_id": task_id, "owner": owner, "acquired_at": timestamp(),
                          "host": socket.gethostname(), "pid": os.getpid()} for name in names)
        event(data, f"{task_id} reserved {', '.join(names)}")


def release(path, task_id, owner, names):
    with transaction(path) as data:
        check_owner(task(data, task_id), owner, required=True)
        for reservation in data.get("resources", []):
            if reservation["name"] in names and (reservation["task_id"], reservation["owner"]) != (task_id, owner):
                raise ValueError(f"Cannot release another task's resource: {reservation['name']}")
        data["resources"] = [r for r in data.get("resources", []) if r["name"] not in names]
        event(data, f"{task_id} released {', '.join(names)}")


def run_reserved(path, task_id, owner, names, command):
    reserve(path, task_id, owner, names)
    child = None
    previous = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        child = subprocess.Popen(command, start_new_session=True)
        return child.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        # Hold the reservation until the foreground process group has stopped.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        release(path, task_id, owner, names)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--file", type=Path, default=Path(os.environ.get("MTG_PROGRESS_PATH", Path(__file__).with_name("progress.json"))))
    commands = root.add_subparsers(dest="cmd", required=True)
    track = commands.add_parser("track")
    track.add_argument("id")
    for field in ("name", "note", "pr", "branch", "owner", "workspace", "handoff"):
        track.add_argument("--" + field)
    track.add_argument("--status", choices=STATUSES)
    for field in ("step", "write-path", "depends-on"):
        track.add_argument("--" + field, action="append")
    result = commands.add_parser("result")
    result.add_argument("name")
    result.add_argument("value")
    commands.add_parser("log").add_argument("msg", nargs="+")
    commands.add_parser("reconcile").add_argument("--repo", required=True)
    commands.add_parser("status").add_argument("--json", action="store_true")
    for verb in ("acquire", "release", "run"):
        cmd = commands.add_parser(verb)
        cmd.add_argument("id")
        cmd.add_argument("--owner", required=True)
        cmd.add_argument("--resource", action="append", required=True)
    return root


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = []
    if "--" in argv:
        split = argv.index("--")
        command, argv = argv[split + 1:], argv[:split]
    args = parser().parse_args(argv)
    path = args.file.expanduser().resolve()
    try:
        if command and args.cmd != "run":
            raise ValueError("Only run accepts a command after --.")
        if args.cmd == "run":
            if not command:
                raise ValueError("run requires -- followed by a foreground command.")
            return run_reserved(path, args.id, args.owner, args.resource, command)
        if args.cmd in ("acquire", "release"):
            (reserve if args.cmd == "acquire" else release)(path, args.id, args.owner, args.resource)
            return 0
        if args.cmd == "reconcile":
            return int(reconcile(path, args.repo))
        if args.cmd == "status":
            data = json.loads(path.read_text())
            if args.json:
                print(json.dumps(data, indent=2))
            else:
                for item in data["tracks"]:
                    remote = item.get("github", {})
                    pr_state = f"#{remote['number']} {remote['state']}" if remote else "PR unverified"
                    if remote.get("isDraft"):
                        pr_state += " draft"
                    if item.get("github_error"):
                        pr_state += " (refresh failed)"
                    print(f"{item['id']}: task={item['status']}; {pr_state}; owner={item.get('owner', 'unclaimed')}; "
                          f"task updated={item.get('updated_at', 'unknown')}; PR observed={remote.get('observed_at', 'never')}")
                for reservation in data.get("resources", []):
                    print(f"BUSY {reservation['name']}: {reservation['task_id']} since {reservation['acquired_at']}")
            return 0
        with transaction(path) as data:
            if args.cmd == "track":
                update_track(data, args)
            elif args.cmd == "result":
                entries = data.setdefault("results", [])
                item = next((item for item in entries if item["name"] == args.name), None)
                if item is None:
                    entries.append({"name": args.name, "value": args.value})
                else:
                    item["value"] = args.value
            else:
                event(data, " ".join(args.msg))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
