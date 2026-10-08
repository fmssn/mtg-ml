"""Versioned, inspectable files. Extraction never approves its own output."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

VERSION = 1
STATUSES = {"pending", "accepted", "rejected"}
VISIBILITY = {"public", "self", "opponent_private", "belief", "hindsight", "unknown"}
COMMENTARY = {"observed_action", "proposed_line", "rejected_alternative", "opponent_prediction", "retrospective_correction", "other"}
KINDS = {"log", "state", "action", "commentary", "gap"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def finite(value, name: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"{name} must be finite")
    return float(value)


def review(status: str = "pending", reviewer: str = "", note: str = "") -> dict:
    return {"status": status, "reviewer": reviewer, "note": note}


def validate_review(value: dict) -> None:
    require(isinstance(value, dict) and value.get("status") in STATUSES, "invalid review status")
    if value["status"] == "accepted":
        require(bool(value.get("reviewer")) and bool(value.get("note")), "accepted records need a reviewer and rationale")


def ref(source_id: str, start: float, end: float, kind: str, locator: str, viewer=None) -> dict:
    return {"source_id": source_id, "start": start, "end": end, "kind": kind, "locator": locator, "viewer": viewer}


def fact(record_id: str, kind: str, value, refs: list[dict], visibility: str = "unknown", available_at: float | None = None, **kw) -> dict:
    return {"id": record_id, "kind": kind, "value": value, "refs": refs, "visibility": visibility,
            "available_at": min(r["start"] for r in refs) if available_at is None else available_at, "review": review(), **kw}


def evidence(source: dict, records: list[dict] | None = None) -> dict:
    return {"format": "EvidenceGame", "version": VERSION, "sources": [source], "records": records or [], "metrics": {}}


def validate_evidence(data: dict) -> dict:
    require(data.get("format") == "EvidenceGame" and data.get("version") == VERSION, "unsupported EvidenceGame version")
    require(isinstance(data.get("sources"), list) and bool(data["sources"]), "sources are required")
    sources = {s["id"]: s for s in data["sources"]}
    require(len(sources) == len(data["sources"]), "duplicate source id")
    require(isinstance(data.get("records"), list), "records must be a list")
    ids = set()
    for record in data["records"]:
        require(bool(record.get("id")) and record["id"] not in ids, "duplicate or empty record id")
        ids.add(record["id"])
        require(record.get("kind") in KINDS and record.get("visibility") in VISIBILITY, "invalid record kind or visibility")
        validate_review(record["review"])
        finite(record["available_at"], "available_at")
        require(bool(record.get("refs")), f"{record['id']}: source references are required")
        for r in record["refs"]:
            require(r.get("source_id") in sources and bool(r.get("locator")), "unknown source or empty locator")
            require(0 <= finite(r["start"], "start") <= finite(r["end"], "end"), "invalid source interval")
            require(r.get("viewer") in (None, 0, 1), "invalid viewer")
        if record["kind"] == "commentary":
            require(record.get("classification", "other") in COMMENTARY, "invalid commentary classification")
    return data


def admissible(record: dict, player: int, decision_time: float, *, action: bool = False) -> bool:
    """Action labels may be observed after the decision; state inputs may not."""
    if record["review"]["status"] != "accepted" or record["kind"] in {"commentary", "gap"}:
        return False
    if record["visibility"] not in {"public", "self"}:
        return False
    if record["visibility"] == "self" and any(r.get("viewer") != player for r in record["refs"]):
        return False
    return action or (record["available_at"] <= decision_time and all(r["end"] <= decision_time for r in record["refs"]))


def group_split(group_id: str, heldout_fraction: float = 0.2) -> str:
    require(0 <= heldout_fraction <= 1, "heldout fraction must be in [0, 1]")
    number = int(hashlib.sha256(group_id.encode()).hexdigest()[:16], 16) / 2**64
    return "test" if number < heldout_fraction else "train"


def read(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write(path: str | Path, data) -> None:
    """Atomic writes; no partial artifact is ever a valid next-stage input."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".expert-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
