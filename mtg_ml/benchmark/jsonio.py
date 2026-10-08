"""Canonical identities and durable, exclusive artifact publication."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"duplicate JSON key {k!r}")
            out[k] = v
        return out

    def constant(s):
        raise ValueError(f"nonfinite JSON number {s}")

    with open(path, encoding="utf-8") as f:
        value = json.load(f, object_pairs_hook=pairs, parse_constant=constant)
    canonical_bytes(value)  # also catches overflowing JSON numbers, e.g. 1e999
    return value


def write_json(path, value) -> None:
    """Publish a complete file without replacing even a concurrent writer."""
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".benchmark-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)  # atomic create-if-absent, never os.replace
    finally:
        os.unlink(tmp)
