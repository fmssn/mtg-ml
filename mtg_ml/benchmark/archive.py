"""Durable release bundles: every artifact with the files it references, the build
environment and a SHA256SUMS that `verify` rechecks, reloading each result and
recomputing each comparison and sensitivity report from the raw rows.

Relative references are preserved byte for byte, so `load_result` works inside the archive.
"""

import importlib.metadata
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from .artifacts import load_manifest, reference, require
from .jsonio import file_digest, read_json, write_json
from .validation import ROOT

SUMS = "SHA256SUMS"
INDEX = "ARCHIVE.json"
SIBLINGS = (".parts", ".failures", ".work")


def _refs(value, parent, found):
    """Every {path, sha256} file reference inside a JSON value, resolved and hash-checked."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str) and set(value) <= {"path", "sha256", "id", "candidate"}:
            found.add(reference(parent, value, "reference"))
        else:
            for v in value.values():
                _refs(v, parent, found)
    elif isinstance(value, list):
        for v in value:
            _refs(v, parent, found)


def closure(paths):
    """The given files plus everything they reference and their sibling .parts/.failures/.work."""
    todo, seen = [Path(p).resolve() for p in paths], set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        require(path.is_file(), str(path), "missing file")
        seen.add(path)
        for suffix in SIBLINGS:
            folder = path.with_name(path.stem + suffix)
            if folder.is_dir():
                todo += [p for p in folder.rglob("*") if p.is_file()]
        if path.name.startswith("turn-limit") or path.name.startswith("games-t"):
            todo += [p for p in path.parent.glob("t*.parts/*") if p.is_file()]
        data = read_json(path) if path.suffix == ".json" else {}
        if isinstance(data, dict) and data.get("format") != "BenchmarkValidation":  # its manifest path is as typed, not relative
            found = set()
            _refs(data, path, found)
            todo += sorted(found)
            if data.get("format") == "BenchmarkPuzzleBundle" and (path.parent / "reviews.json").is_file():
                todo.append(path.parent / "reviews.json")
    return seen


def _run(*argv, cwd=None):
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
        return (proc.stdout + proc.stderr).strip()
    except OSError as e:
        return f"unavailable: {e}"


def _environment(folder, revision):
    folder.mkdir()
    freeze = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata["Name"])
    (folder / "pip-freeze.txt").write_text("\n".join(freeze) + "\n")
    (folder / "toolchain.txt").write_text(f"rustc: {_run('rustc', '-V')}\ncargo: {_run('cargo', '-V')}\n")
    try:
        from .validation import native_build
        build = native_build()
    except ValueError as e:
        build = {"unavailable": str(e)}
    write_json(folder / "native-build.json", build)
    (folder / "git.txt").write_text(f"HEAD: {_run('git', 'rev-parse', 'HEAD', cwd=ROOT)}\nfreeze: {revision}\n"
                                    f"status:\n{_run('git', 'status', '--porcelain', cwd=ROOT)}\n")
    with open(folder / "source.tar.gz", "wb") as f:
        subprocess.run(["git", "archive", "--format=tar.gz", revision], cwd=ROOT, stdout=f, check=True)


def _sums(root):
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.relative_to(root).as_posix() != SUMS:
            lines.append(f"{file_digest(p)}  {p.relative_to(root).as_posix()}")
    return lines


def build(out, *, manifest, results=(), comparisons=(), calibration=None, sensitivity=None, validations=()):
    """Create a new archive directory; refuses to touch an existing one."""
    out = Path(out).resolve()
    require(not out.exists(), "out", f"{out} exists; archives are append-only")
    roles = {"manifest": [manifest], "results": list(results), "comparisons": list(comparisons),
             "calibration": [calibration] if calibration else [], "sensitivity": [sensitivity] if sensitivity else [], "validation": list(validations)}
    named = [Path(p).resolve() for group in roles.values() for p in group]
    files = closure(named)
    revision = load_manifest(manifest).data["freeze"]["code_revision"]
    base = Path(os.path.commonpath([str(p.parent) for p in files]))
    out.mkdir(parents=True)
    for p in sorted(files):
        target = out / "artifacts" / p.relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    _environment(out / "environment", revision)
    commands = []
    for p in named:
        runtime = read_json(p).get("runtime") if p.suffix == ".json" else None
        if isinstance(runtime, dict) and runtime.get("command"):
            commands.append(f"# {p.name}\n{shlex.join(runtime['command'])}")
    (out / "commands.txt").write_text("\n".join(commands) + "\n")
    write_json(out / INDEX, {"format": "BenchmarkArchive", "version": 1, "code_revision": revision,
                             "roles": {k: [(Path("artifacts") / Path(p).resolve().relative_to(base)).as_posix() for p in v] for k, v in roles.items()}})
    (out / SUMS).write_text("\n".join(_sums(out)) + "\n")
    return out


def verify(path):
    """Recheck hashes, then reload every result, comparison and sensitivity report. Raises ValueError listing every problem."""
    from .release import verify_sensitivity
    from .results import load_comparison, load_result
    root = Path(path).resolve()
    problems = []
    try:
        listed = dict(line.split("  ", 1)[::-1] for line in (root / SUMS).read_text().splitlines() if line)
    except OSError as e:
        raise ValueError(f"{root}: {e}") from e
    actual = {line.split("  ", 1)[1] for line in _sums(root)}
    problems += [f"unlisted file {p}" for p in sorted(actual - set(listed))]
    for rel, digest in sorted(listed.items()):
        target = root / rel
        if not target.is_file():
            problems.append(f"missing file {rel}")
        elif file_digest(target) != digest:
            problems.append(f"modified file {rel}")
    if not problems:
        for p in sorted((root / "artifacts").rglob("*.json")):
            try:
                fmt = read_json(p).get("format")
                if fmt == "BenchmarkResult":
                    load_result(p)
                elif fmt == "BenchmarkComparison":
                    load_comparison(p)
                elif fmt == "BenchmarkTurnLimitSensitivity":
                    verify_sensitivity(p)
            except (ValueError, KeyError, TypeError, OSError) as e:
                problems.append(f"{p.relative_to(root)}: {e}")
    if problems:
        raise ValueError("; ".join(problems))
    return {"files": len(listed), "root": str(root)}
