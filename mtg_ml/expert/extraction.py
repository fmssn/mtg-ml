"""Deterministic text recovery and alignment. Hypotheses remain pending."""

from __future__ import annotations

from difflib import SequenceMatcher
import html
import re

from .artifacts import fact, ref


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", "", text))).strip()


def same_line(a: str, b: str) -> bool:
    a, b = clean(a).casefold(), clean(b).casefold()
    # Fuzzy punctuation/OCR matching is useful for scrolling; changing
    # numbers or card names is material, so never fuzz those tokens away.
    if re.findall(r"\d+", a) != re.findall(r"\d+", b):
        return False
    return a == b or (len(a) > 20 and SequenceMatcher(None, a, b).ratio() >= 0.96)


def logical_log_lines(lines: list[str]) -> list[str]:
    """Join wrapped MTGO timestamped sentences, keeping separate events."""
    stamp = re.compile(r"^\d{1,2}:\d{2}\s*[AP]M\s*:", re.I)
    if not any(stamp.match(clean(line)) for line in lines):
        return lines  # preserve uncertain OCR for review
    result = []
    for line in lines:
        line = clean(line)
        if stamp.match(line):
            result.append(line)
        elif result:
            result[-1] += " " + line
    return result


def merge_log(windows: list[dict], source_id: str) -> list[dict]:
    """Append longest suffix/prefix overlap, keeping repeated line occurrences.

    A repeated unchanged viewport cannot establish that an identical event
    happened again. Cuts and windows without overlap are explicit gaps.
    """
    records, previous = [], []
    for window in windows:
        lines = [clean(x) for x in window["lines"] if clean(x)]
        refs = [ref(source_id, window["time"], window["time"], "frame", window["frame"])]
        overlap = 0
        if previous and not window.get("cut", False):
            for n in range(min(len(previous), len(lines)), 0, -1):
                if all(same_line(a, b) for a, b in zip(previous[-n:], lines[:n])):
                    overlap = n
                    break
        if window.get("occluded", False) or window.get("cut", False) or (previous and lines and not overlap):
            reason = "occluded" if window.get("occluded") else "cut" if window.get("cut") else "missing_log_overlap"
            records.append(fact(f"gap-{len(records)}", "gap", {"reason": reason}, refs))
        for line in lines[overlap:]:
            records.append(fact(f"log-{len(records)}", "log", {"text": line}, refs, visibility="public"))
        previous = lines  # empty/occluded windows break the continuity chain
    return records


def seconds(timestamp: str) -> float:
    value = 0.0
    for part in timestamp.replace(",", ".").split(":"):
        value = value * 60 + float(part)
    return value


def parse_vtt(text: str, source_id: str, locator: str, start: float = 0, end: float = float("inf")) -> list[dict]:
    """YouTube rolling cues repeat words: retain new words, not duplicates."""
    records, previous = [], []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = block.splitlines()
        at = next((i for i, line in enumerate(lines) if " --> " in line), None)
        if at is None:
            continue
        left, right = lines[at].split(" --> ", 1)
        a, b = seconds(left.strip()), seconds(right.split()[0])
        if b < start or a > end:
            continue
        words = clean(" ".join(lines[at + 1:])).split()
        n = next((k for k in range(min(len(previous), len(words)), 0, -1) if previous[-k:] == words[:k]), 0)
        new = " ".join(words[n:])
        if new:
            records.append(fact(f"caption-{len(records)}", "commentary", {"text": new},
                                [ref(source_id, max(a, start), min(b, end), "caption", f"{locator}#cue-{len(records)}")],
                                classification="other"))
        previous = words
    return records


def card_mentions(text: str, card_names: list[str], aliases: dict[str, str] | None = None) -> list[str]:
    names = {n.casefold(): n for n in card_names}
    names.update({k.casefold(): v for k, v in (aliases or {}).items()})
    out = set()
    for name, canonical in names.items():
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE):
            out.add(canonical)
    return sorted(out)


def classify(text: str) -> str:
    """A review aid, never a strategic label or ground truth."""
    text = text.casefold()
    if re.search(r"should have|should've|misplay|that was wrong", text):
        return "retrospective_correction"
    if re.search(r"they (might|probably|could|must) have|play around|opponent may", text):
        return "opponent_prediction"
    if re.search(r"don't want to|do not want to|instead of|rather than", text):
        return "rejected_alternative"
    if re.search(r"we (could|can|should)|i (could|can|should)|going to|let's", text):
        return "proposed_line"
    return "other"


def align(records: list[dict], card_names: list[str], aliases: dict[str, str] | None = None, window: float = 15) -> list[dict]:
    """Rank evidence links by names + temporal distance; retain alternatives."""
    actions = [r for r in records if r["kind"] in {"log", "action"}]
    for r in records:
        if r["kind"] != "commentary":
            continue
        text = r["value"].get("text", "")
        r["classification"] = classify(text)
        r["cards"] = card_mentions(text, card_names, aliases)
        if r["classification"] == "opponent_prediction":
            r["visibility"] = "belief"
        elif r["classification"] == "retrospective_correction":
            r["visibility"] = "hindsight"
        candidates = []
        t = r["refs"][0]["start"]
        for a in actions:
            distance = abs(a["refs"][0]["start"] - t)
            if distance > window or a["refs"][0]["source_id"] != r["refs"][0]["source_id"]:
                continue
            cards = card_mentions(a["value"].get("text", ""), card_names, aliases)
            shared = sorted(set(cards) & set(r["cards"]))
            score = len(shared) * 2 + 1 - distance / window
            candidates.append({"record_id": a["id"], "score": round(score, 4), "shared_cards": shared})
        r["alignment_candidates"] = sorted(candidates, key=lambda x: (-x["score"], x["record_id"]))[:5]
    return records


def event_metrics(predicted: list[str], gold: list[str]) -> dict:
    """Ordered, occurrence-sensitive matching; missing repetitions count."""
    a, b = [clean(x) for x in predicted], [clean(x) for x in gold]
    matched = sum(m.size for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    precision = matched / len(a) if a else float(not b)
    recall = matched / len(b) if b else 1.0
    return {"predicted": len(a), "gold": len(b), "matched": matched, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
