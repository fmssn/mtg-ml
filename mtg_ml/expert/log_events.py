"""MTGO log grammar and occurrence alignment; interpretations are review proposals."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata

from ..engine.cards import CARDS, TOKENS
from .artifacts import require, review


def folded(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text).casefold() if not unicodedata.combining(c))


def parse_event(text, players, aliases=None):
    """Recognize literal actors/cards, including joined OCR words, without fuzzy amounts."""
    text = re.sub(r"^\s*\d{1,2}:\d{2}\s*[AP]M\s*:\s*", "", text, flags=re.I).strip().rstrip(".")
    for alias, canonical in (aliases or {}).items():
        text = text.replace(alias, canonical)
    names = sorted(set(CARDS) | set(TOKENS), key=lambda n: (-len(n), n))
    # MTGO appends 'Token(s)' to the engine's token names.
    cards = []
    occupied = set()
    for name in names:
        for match in re.finditer(re.escape(folded(name)), folded(text)):
            span = set(range(match.start(), match.end()))
            if not span & occupied:
                cards.append((match.start(), name))
                occupied |= span
    cards = [n for _, n in sorted(cards)]
    actor = next(
        (
            int(seat)
            for name, seat in sorted(players.items(), key=lambda x: -len(x[0]))
            if folded(text).startswith(folded(name))
        ),
        None,
    )
    body = text
    if actor is not None:
        name = next(
            name for name, seat in players.items() if int(seat) == actor and folded(text).startswith(folded(name))
        )
        body = text[len(name) :].strip()
    lower = folded(body)
    event = {"type": "unsupported", "player": actor, "cards": cards}
    turn = re.match(r"^Turn\s*(\d+)\s*:\s*(.+)$", text, re.I)
    if turn:
        event.update(type="turn", number=int(turn[1]), player=players.get(turn[2].strip()))
    elif lower.startswith("casts"):
        event.update(type="cast", card=cards[0] if cards else None, targets=cards[1:] if "targeting" in lower else [])
        target = next(
            (
                int(s)
                for n, s in players.items()
                if "targeting" in lower and folded(n) in lower.split("targeting", 1)[1]
            ),
            None,
        )
        if target is not None:
            event["target_player"] = target
        event["method"] = "escape" if "escape cost" in lower else "flashback" if "flashback" in lower else "normal"
        mode = re.search(r'choosing\s*"([^"]+)"', body)
        if mode:
            event["mode"] = mode[1]
    elif lower.startswith("plays"):
        event.update(type="play_land", card=cards[0] if cards else None)
    elif lower.startswith("activates"):
        event.update(type="activate", card=cards[0] if cards else None)
    elif lower.startswith("cycles"):
        event.update(type="cycle", card=cards[0] if cards else None)
    elif lower.startswith("draws"):
        count = re.match(r"draws\s*(a|one|two|three|\d+)\s*cards?", lower)
        event.update(
            type="draw",
            count={"a": 1, "one": 1, "two": 2, "three": 3}.get(
                count[1], int(count[1]) if count and count[1].isdigit() else 1
            )
            if count
            else 1,
        )
    elif lower.startswith("discards"):
        event.update(type="discard")
    elif lower.startswith("mills"):
        event.update(type="mill")
    elif lower.startswith("exiles"):
        event.update(type="exile", cards=[n for n in cards if folded(n) in lower.split(" with ", 1)[0]])
    elif lower.startswith("reveals"):
        event.update(type="reveal", cards=[n for n in cards if folded(n) in lower.split(":", 1)[-1]])
    elif lower.startswith("shuffles"):
        event.update(type="shuffle")
    elif lower.startswith("sacrifices"):
        event.update(type="sacrifice", card=cards[0] if cards else None)
    elif lower.startswith("scrys") or lower.startswith("scries"):
        event.update(type="scry")
    elif "is being attacked by" in lower:
        event.update(type="attack", player=None if actor is None else 1 - actor)
    elif "blocks" in lower:
        event.update(type="block", card=cards[0] if cards else None, targets=cards[1:])
    elif "puts a triggered ability" in lower:
        event.update(type="trigger", card=cards[0] if cards else None)
    elif "puts a +1/+1 counter" in lower:
        event.update(type="counter", card=cards[-1] if cards else None)
    elif "creates" in lower:
        event.update(type="token")
    elif "returned to" in lower and "hand" in lower:
        recipient = next((int(s) for n, s in players.items() if folded(n) in lower), actor)
        event.update(type="return_hand", player=recipient)
    elif "chooses to use" in lower or "declines to use" in lower:
        event.update(type="choice", yes="chooses to use" in lower)
    elif "chooses" in lower and ("bottom" in lower or "top" in lower):
        event.update(type="order_choice")
    elif "begins the game with" in lower or "mulligans to" in lower or "begins the game with" in folded(text):
        event.update(type="opening")
    elif "chooses to play first" in lower:
        event.update(type="starting_player")
    elif "skips their draw" in lower:
        event.update(type="skip_draw")
    elif "has conceded" in lower:
        event.update(type="concede")
    elif "wins the match" in lower or "wins the game" in lower or "match tied" in lower or "leads the match" in lower:
        event.update(type="result")
    elif "joined the game" in lower or "rolled a" in lower or "lost connection" in lower:
        event.update(type="lobby")
    elif "explores" in lower:
        event.update(type="explore", card=cards[0] if cards else None)
    if event["type"] == "token":
        count = re.search(r"creates\s+(a|an|one|two|three|\d+)\b", lower)
        if count:
            event["count"] = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}.get(
                count[1], int(count[1]) if count[1].isdigit() else 1
            )
    if event["type"] == "counter":
        event["count"] = 1
    if event["type"] in {"cast", "play_land", "activate", "cycle"} and not event.get("card"):
        event["type"] = "unsupported"
    return event


def signature(event):
    return tuple((k, repr(v)) for k, v in sorted(event.items()))


def normalize_events(evidence, spec):
    """Keep occurrence identity. Re-shown blocks need ordered context, never a set of texts."""
    result, history = [], {}
    overrides = spec.get("overrides", {})
    for record in evidence["records"]:
        if record["kind"] not in {"log", "gap"}:
            continue
        refs = [r for r in record["refs"] if r["source_id"] == spec["source_id"]]
        if not refs:
            continue
        t = record["available_at"]
        games = [g for g in spec["games"] if g["start"] <= t < g["end"]]
        game = games[0]["id"] if games else None
        item = {
            "id": record["id"],
            "game_id": game,
            "available_at": t,
            "refs": refs,
            "text": record["value"].get("text", ""),
            "review": review(),
            "evidence_review": record["review"],
            "status": "gap" if record["kind"] == "gap" else "outside_game" if game is None else "proposed",
        }
        item["event"] = (
            {"type": "gap", **record["value"]}
            if record["kind"] == "gap"
            else parse_event(item["text"], spec["players"], spec.get("aliases"))
        )
        if record["id"] in overrides:
            patch = overrides[record["id"]]
            require(
                set(patch)
                <= {"event", "duplicate_of", "review", "status", "text", "game_id", "audit_refs", "recovery"},
                "unknown event correction field",
            )
            if "text" in patch:
                item["event"] = parse_event(patch["text"], spec["players"], spec.get("aliases"))
            item.update(patch)
            if item.get("duplicate_of"):
                item["status"] = "duplicate"
        game = item["game_id"]
        require(game is None or game in {g["id"] for g in spec["games"]}, "correction names unknown game")
        if item["event"]["type"] == "turn" and game and item["event"].get("player") is not None:
            starting = next(g["starting_player"] for g in spec["games"] if g["id"] == game)
            item["event"]["engine_turn"] = (item["event"]["number"] - 1) * 2 + 1 + (item["event"]["player"] != starting)
        result.append(item)
        history[item["id"]] = item
    # A clock/semantic sequence recurs after scrolling back. Suggest only aligned blocks of >=3.
    for game in spec["games"]:
        rows = [r for r in result if r["game_id"] == game["id"] and r["event"]["type"] != "gap"]
        previous = []
        batches = []
        for row in rows:
            if not batches or row["available_at"] != batches[-1][0]["available_at"]:
                batches.append([])
            batches[-1].append(row)
        for batch in batches:
            before = [signature(r["event"]) for r in previous]
            now = [signature(r["event"]) for r in batch]
            blocks = SequenceMatcher(None, before, now, autojunk=False).get_matching_blocks()
            for block in blocks:
                if block.size < 3:
                    continue
                for a, b in zip(previous[block.a : block.a + block.size], batch[block.b : block.b + block.size]):
                    if a["available_at"] >= b["available_at"]:
                        continue
                    # Clock is additional evidence for a re-shown occurrence, not event timing.
                    clock = lambda r: re.match(r"^\s*(\d{1,2}:\d{2}\s*[AP]M)", r["text"], re.I)
                    ca, cb = clock(a), clock(b)
                    if ca and cb and ca[1] == cb[1] and b["id"] not in overrides:
                        b["duplicate_candidate"] = a["id"]
            previous.extend(r for r in batch if r.get("status") != "duplicate" and not r.get("duplicate_candidate"))
    for item in result:
        if item.get("duplicate_of"):
            target = history.get(item["duplicate_of"])
            require(
                target is not None
                and target["game_id"] == item["game_id"]
                and target["available_at"] < item["available_at"],
                "invalid duplicate occurrence reference",
            )
    return result
