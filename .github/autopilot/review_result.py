"""Validate the reviewer contract; completion is separate from process success."""

import json
import re
from pathlib import Path


class InvalidResult(ValueError):
    pass


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def parse_result(envelope):
    if not isinstance(envelope, dict):
        raise InvalidResult("model produced no result envelope")
    if envelope.get("is_error") or envelope.get("subtype", "success") != "success":
        raise InvalidResult(f"model ended with {envelope.get('subtype', 'an error')}")
    result = envelope.get("structured_output")
    if result is None:
        text = envelope.get("result", "")
        if not isinstance(text, str):
            raise InvalidResult("model result is not text")
        blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.S)
        try:
            result = json.loads(blocks[-1] if blocks else text)
        except (ValueError, IndexError) as exc:
            raise InvalidResult("model produced no valid JSON verdict") from exc
    if not isinstance(result, dict):
        raise InvalidResult("verdict must be an object")
    if result.get("verdict") not in {"clean", "fixed", "escalate", "incomplete"}:
        raise InvalidResult("unknown verdict")
    if not isinstance(result.get("summary"), str) or not result["summary"].strip():
        raise InvalidResult("missing summary")
    if not isinstance(result.get("findings"), list):
        raise InvalidResult("findings must be a list")
    for finding in result["findings"]:
        if not isinstance(finding, dict) or not isinstance(finding.get("file"), str) or not finding["file"]:
            raise InvalidResult("finding must name a file")
        if type(finding.get("line")) is not int or finding["line"] < 1:
            raise InvalidResult("finding must name a positive line number")
        if not isinstance(finding.get("issue"), str) or not finding["issue"].strip():
            raise InvalidResult("finding must explain the defect")
        if finding.get("status") not in {"fixed", "open"}:
            raise InvalidResult("unknown finding status")
    if not isinstance(result.get("escalation_reason", ""), str):
        raise InvalidResult("escalation_reason must be text")
    for field in ("reviewed_files", "pending_checks"):
        if field in result and (not isinstance(result[field], list) or any(not isinstance(x, str) for x in result[field])):
            raise InvalidResult(f"{field} must be a list of strings")
    if result["verdict"] == "clean" and result["findings"]:
        raise InvalidResult("clean verdict contains findings")
    if result["verdict"] in {"clean", "fixed"} and any(f["status"] == "open" for f in result["findings"]):
        raise InvalidResult("completed verdict contains unresolved findings")
    if result["verdict"] == "escalate" and not (result["findings"] or result.get("escalation_reason", "").strip()):
        raise InvalidResult("escalation needs an evidenced blocker or a specific decision")
    return result


def completion_problems(result, manifest=None):
    problems = list(result.get("pending_checks", []))
    if result["verdict"] == "incomplete":
        problems.insert(0, result.get("escalation_reason") or "review is unfinished")
    # Old prompts/results remain usable when no new context manifest exists.
    if manifest and manifest.get("version", 1) >= 2:
        if "reviewed_files" not in result or "pending_checks" not in result:
            problems.append("review omitted coverage or pending-check lists")
        required = {f["path"] for f in manifest["files"] if not f.get("excluded")}
        missing = required - set(result.get("reviewed_files", []))
        if missing:
            problems.append("unreviewed files: " + ", ".join(sorted(missing)))
    return problems
