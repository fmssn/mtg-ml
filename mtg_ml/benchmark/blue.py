"""Deterministic preboard Mono Blue Terror specialist, own-list-hidden-opponent-v1.

Rules use detached data only. Combat optimization concerns the current public
combat, not engine forks, hidden-world sampling or game-tree search. Unrecognized
semantics fail visibly instead of selecting an arbitrary legal option.
"""

from collections import Counter, deque
from itertools import accumulate
from functools import lru_cache

from ..engine.mana import ManaCost, RemainingCost, can_pay, pool_units
from .adapters import AgentMetadata, ScriptedAdapter
from .artifacts import FAIR
from .jsonio import digest
from .views import freeze, thaw

BOT_ID = "benchmark-blue@1"
RULES_REVISION = 1
PARAMETERS = freeze({
    "desired_lands": 4, "one_land_cantrips": 2, "counter_min_value": 2.5,
    "spike_min_value": 1.5, "known_top_keep_value": 3.5,
    "reserve_blue": 2, "cheap_threat_cost": 2,
    "cards": {"Counterspell": 5.0, "Force Spike": 2.5, "Tolarian Terror": 4.5,
              "Cryptic Serpent": 4.0, "Delver of Secrets": 3.5, "Brainstorm": 3.0,
              "Ponder": 3.0, "Thought Scour": 2.5, "Mental Note": 2.3,
              "Lorien Revealed": 2.5, "Deem Inferior": 3.0, "Sleep of the Dead": 2.0,
              "Plunder the Trollshaws": 2.0},
    "spells": {"Writhing Chrysalis": 4.5, "Refurbished Familiar": 3.5,
               "Gixian Infiltrator": 2.5, "Krark-Clan Shaman": 3.0,
               "Nyxborn Hydra": 3.0, "Fanatical Offering": 2.5,
               "Eviscerator's Insight": 2.5, "Makeshift Munitions": 3.0,
               "Cleansing Wildfire": 1.5, "Toxin Analysis": 1.0,
               "Ichor Wellspring": 0.5, "Lembas": 0.5, "Nihil Spellbomb": 1.0,
               "Tolarian Terror": 4.5, "Cryptic Serpent": 4.0,
               "Delver of Secrets": 3.0, "Lorien Revealed": 3.0,
               "Brainstorm": 1.0, "Ponder": 1.0, "Thought Scour": 0.5,
               "Mental Note": 0.5, "Plunder the Trollshaws": 2.0},
})
THREATS = {"Tolarian Terror", "Cryptic Serpent"}
CANTRIPS = {"Brainstorm", "Ponder", "Thought Scour", "Mental Note"}
COUNTERS = {"Counterspell", "Force Spike", "Dispel", "Annul", "Envelop",
            "Blue Elemental Blast", "Hydroblast", "Red Elemental Blast", "Pyroblast"}
NEG = -1e9


def register_blue(registry, source_revision):
    """Register a frozen rule identity in the caller's registry; no global edits."""
    parameters = thaw(PARAMETERS)
    registry.register(AgentMetadata(BOT_ID, "mono_blue_terror", source_revision,
                                   RULES_REVISION, digest(parameters), FAIR), BenchmarkBlue, parameters)


def blue_factory(seat, mode):
    if mode not in {"sampled", "greedy"}:
        raise ValueError(f"unsupported mode: {mode}")
    return ScriptedAdapter(BenchmarkBlue(), seat)


def creatures(view, side):
    return [c for c in view.state["battlefield"] if c["controller"] == side and "Creature" in c["types"]]


def permanent(view, oid):
    return next((c for c in view.state["battlefield"] if c["oid"] == oid), None)


def keyword(card, name):
    return name in card["keywords"]


def creature_value(card):
    return max(0, card["power"]) * 1.5 + max(0, card["toughness"]) * .5 + (1.5 if keyword(card, "flying") else 0)


def blocks(blocker, attacker):
    return not keyword(attacker, "unblockable") and (not keyword(attacker, "flying") or
                                                     keyword(blocker, "flying") or keyword(blocker, "reach"))


def lethal(source, target):
    return 1 if keyword(source, "deathtouch") and source["power"] > 0 else max(0, target["toughness"] - target["damage"])


def mana_units(view, side):
    units = list(pool_units(dict(view.state[side]["pool"])))
    for card in view.state["battlefield"]:
        if card["controller"] != side:
            continue
        for ability in view.cards[card["name"]]["abilities"]:
            if not ability["mana"] or ability["zone"] != "battlefield":
                continue
            if ability["tap"] and (card["tapped"] or ("Creature" in card["types"] and card["sick"])):
                continue
            units.append(tuple(ability["mana"]))
            break
    return units


def available(view, side="self"):
    return len(mana_units(view, side))


def blue_available(view):
    return sum("U" in unit for unit in mana_units(view, "self"))


def can_afford(view, cost, side="self"):
    cost = ManaCost.parse(cost)
    return can_pay(RemainingCost.of(cost), mana_units(view, side))


@lru_cache(maxsize=512)
def _prevented(powers, legal):
    """Maximum damage stopped by one blocker per attacker, via assignment DP.

    For natural Blue attackers (no trample) extra blockers cannot stop extra
    player damage. Additional blockers still matter to the separate trade rule.
    Duplicate blocking capabilities are processed only up to attacker count.
    """
    dp = {0: 0}
    for mask, count in Counter(legal).items():
        for _ in range(min(count, len(powers))):
            new = dict(dp)
            for used, score in dp.items():
                free = mask & ~used
                while free:
                    bit = free & -free
                    free -= bit
                    index = bit.bit_length() - 1
                    key = used | bit
                    new[key] = max(new.get(key, -1), score + powers[index])
            dp = new
    return max(dp.values(), default=0)


def damage_floor(attackers, defenders):
    """Guaranteed damage; optimize blocking, including cumulative trample.

    Ordinary Blue attacks have at most twelve bodies. Unusually wide boards or
    multiple tramplers use a conservative prevention upper bound, never a false
    guaranteed-lethal claim. No legal actions or entities are truncated.
    """
    if not attackers:
        return 0
    powers = tuple(max(0, c["power"]) for c in attackers)
    tramplers = [i for i, a in enumerate(attackers) if keyword(a, "trample")]
    if len(attackers) > 12 or len(tramplers) > 1:
        maximum = sum(max((min(powers[i], lethal(a, b)) if keyword(a, "trample") else powers[i]
                           for i, a in enumerate(attackers) if blocks(b, a)), default=0) for b in defenders)
        return max(0, sum(powers) - maximum)
    if tramplers:
        t = tramplers[0]
        ordinary = [a for i, a in enumerate(attackers) if i != t]
        ps = tuple(max(0, a["power"]) for a in ordinary)
        choices = tuple((sum(1 << i for i, a in enumerate(ordinary) if blocks(b, a)),
                         lethal(attackers[t], b) if blocks(b, attackers[t]) else 0) for b in defenders)
        return sum(powers) - _trample_prevented(ps, powers[t], choices)
    legal = tuple(sum(1 << i for i, a in enumerate(attackers) if blocks(b, a)) for b in defenders)
    return sum(powers) - _prevented(powers, legal)


@lru_cache(maxsize=512)
def _trample_prevented(powers, trample_power, choices):
    # State is (which ordinary attacks were blocked, trample damage stopped).
    dp = {(0, 0): 0}
    for legal, capacity in choices:
        out = dict(dp)
        for (mask, stopped), value in dp.items():
            if capacity:
                new = min(trample_power, stopped + capacity)
                out[mask, new] = max(out.get((mask, new), -1), value + new - stopped)
            free = legal & ~mask
            while free:
                bit = free & -free
                free -= bit
                key = mask | bit, stopped
                out[key] = max(out.get(key, -1), value + powers[bit.bit_length() - 1])
        dp = out
    return max(dp.values())


def _suffix_scores(weights, lethals, remaining, start, force_lethal):
    """O(blockers * power) suffix DP for complete/sequential damage parity."""
    dp = [0.0] + [-float("inf")] * remaining
    for i in reversed(range(start, len(weights))):
        need = lethals[i]
        prefix = list(accumulate(dp, max))
        window, out = deque(), []
        for n in range(remaining + 1):
            kill = prefix[n - need] + weights[i] if n >= need else -float("inf")
            while window and dp[window[-1]] <= dp[n]:
                window.pop()
            window.append(n)
            while window and window[0] <= n - need:
                window.popleft()
            spare = dp[window[0]] if window and not force_lethal else -float("inf")
            out.append(max(kill, spare))
        dp = out
    return dp


class BenchmarkBlue:
    name = BOT_ID

    def __init__(self):
        self.own_deck = None

    def reset(self, own_deck):
        self.own_deck = freeze(dict(own_deck))

    def observe(self, visible_event):
        # The foundation maintains all known cards and current stack references.
        # No redundant inferred history is required by a reset-mode fixture.
        if visible_event.actor not in {"self", "opponent"}:
            raise ValueError("unsupported event actor")

    def choose(self, view, actions):
        if self.own_deck is None:
            raise ValueError("reset required")
        if view.own_deck != self.own_deck:
            raise ValueError("own-list changed without reset")
        if not actions or any(a.index != i or a.kind != actions[0].kind for i, a in enumerate(actions)):
            raise ValueError("invalid current action sequence")
        handler = getattr(self, "_" + actions[0].kind, None)
        if handler is None:
            raise ValueError(f"unsupported decision: {actions[0].kind}")
        scores = handler(view, actions)
        return max(range(len(actions)), key=scores.__getitem__)

    def _bad(self, action):
        raise ValueError(f"unsupported {action.kind} action: {action.key}")

    def _lands(self, view):
        return sum(c["controller"] == "self" and "Land" in c["types"] for c in view.state["battlefield"])

    def _value(self, view, name):
        if name is None:
            return NEG
        if name == "Island":
            have = self._lands(view) + view.state["self"]["hand"].count("Island")
            return 5.0 if have <= 2 else 3.8 if have < PARAMETERS["desired_lands"] else .5
        value = PARAMETERS["cards"].get(name, 1.0)
        if name == "Force Spike" and view.state["turn"] > 8:
            value = 1.0
        if name in THREATS and view.state["self"]["hand"].count(name) > 1:
            value -= 1.0
        if name == "Delver of Secrets" and view.state["turn"] > 8:
            value -= 1.0
        return value

    def _main(self, view):
        return view.state["active"] == "self" and view.state["step"] in {"main1", "main2"} and not view.context["stack"]

    def _end(self, view):
        return view.state["active"] == "opponent" and view.state["step"] == "end" and not view.context["stack"]

    def _fuel(self, view):
        return sum(bool(set(view.cards[n]["types"]) & {"Instant", "Sorcery"}) for n in view.state["self"]["graveyard"])

    def _mill_loss(self, view):
        return max((self._value(view, name) for pos, name in view.state["self"]["library_known"] if pos < 2), default=0)

    def _stack(self, view, sid):
        return next((s for s in view.context["stack"] if s["sid"] == sid), None)

    def _handled(self, view, item):
        # Only definite successful counters count. A payable Force Spike does
        # not settle the spell. A counter underneath its target cannot resolve
        # in time. Ward taxes are cumulative and tied to actual spell IDs.
        stack = view.context["stack"]
        pos = next(i for i, s in enumerate(stack) if s["sid"] == item["sid"])
        tax = 0
        for s in stack[pos + 1:]:
            if s.get("ward", {}).get("sid") == item["sid"]:
                tax += s["ward"]["amount"]
            if s["name"] in COUNTERS and any(t.get("sid") == item["sid"] for t in s["targets"]):
                # A still-higher opposing counter targeting this counter means
                # it is not certain to resolve; preserve our backup counter.
                disrupted = any(t.get("sid") == s["sid"] for higher in stack[stack.index(s) + 1:]
                                if higher["name"] in COUNTERS for t in higher["targets"])
                if not disrupted and (s["name"] != "Force Spike" or available(view, item["controller"]) < 1):
                    return True
        return tax > available(view, item["controller"])

    def _spell_value(self, view, item):
        if item["kind"] != "spell" or item["controller"] != "opponent" or self._handled(view, item):
            return NEG
        value = PARAMETERS["spells"].get(item["name"], 1.0)
        for target in item["targets"]:
            if "oid" in target:
                c = permanent(view, target["oid"])
                if c is not None and c["controller"] == "self":
                    # Milling/tapping does not deserve a removal-level rating.
                    if item["name"] == "Sleep of the Dead":
                        value = max(value, 2.5 if c["power"] >= 3 else 1.0)
                    else:
                        value = max(value, creature_value(c) + 1 if "Creature" in c["types"] else 3.0)
            elif "sid" in target:
                s = self._stack(view, target["sid"])
                if s is not None and s["controller"] == "self":
                    value = max(value, PARAMETERS["cards"].get(s["name"], 3.0) + 1)
        if item["name"] == "Nyxborn Hydra":
            value += item["x"] * .7
        return value

    def _counter_target(self, view, spike=False):
        candidates = [s for s in view.context["stack"] if s["kind"] == "spell" and s["controller"] == "opponent"]
        if spike:
            candidates = [s for s in candidates if self._spike_live(view, s)]
        return max(candidates, key=lambda s: self._spell_value(view, s), default=None)

    def _spike_live(self, view, item):
        return available(view, item["controller"]) < 1 + sum(
            w["ward"]["amount"] for w in view.context["stack"] if w.get("ward", {}).get("sid") == item["sid"])

    def _reserve(self, view, cost):
        protecting = any(c["power"] >= 3 or c["name"] == "Delver of Secrets" for c in creatures(view, "self"))
        # Two mana on an empty enemy hand is not automatically productive.
        return (protecting and view.state["opponent"]["hand_count"] > 0 and
                "Counterspell" in view.state["self"]["hand"] and
                blue_available(view) >= 2 and available(view) - cost < PARAMETERS["reserve_blue"])

    def _tempo(self, view, card, sleep=False, escape=False):
        if card["controller"] != "opponent" or (sleep and "Creature" not in card["types"]):
            return NEG
        if sleep and card["tapped"] and card["skip_untap"] > 0:
            return NEG
        mine, theirs = creatures(view, "self"), creatures(view, "opponent")
        ready = [c for c in mine if not c["tapped"] and not c["sick"]]
        defenders = [c for c in theirs if not c["tapped"] and c["oid"] != card["oid"]]
        before = damage_floor(ready, [c for c in theirs if not c["tapped"]])
        after = damage_floor(ready, defenders)
        # Lifelink blockers can increase the defender's life during damage.
        life = view.state["opponent"]["life"] + sum(max(0, c["power"]) for c in defenders if keyword(c, "lifelink"))
        if view.state["step"] == "main1" and after >= life and before < life:
            return 95.0
        upcoming = [c for c in theirs if c["skip_untap"] == 0 and c["oid"] != card["oid"]]
        blockers = [c for c in mine if not c["tapped"]]
        danger_before = damage_floor([c for c in theirs if c["skip_untap"] == 0], blockers)
        danger_after = damage_floor(upcoming, blockers)
        if danger_before >= view.state["self"]["life"] and danger_after < view.state["self"]["life"]:
            return 90.0
        if "Creature" not in card["types"]:
            return 12.0 if card["name"] in {"Makeshift Munitions", "Nihil Spellbomb"} else NEG
        gain = max(0, after - before) if view.state["step"] == "main1" else 0
        if sleep:
            if not gain:
                return NEG
            # Escaping before a threat enters can make that threat unaffordable.
            if escape and any(n in THREATS for n in view.state["self"]["hand"]) and self._fuel(view) <= 5:
                return NEG
            return 15.0 + gain
        return 18.0 + gain + creature_value(card) * .1 if gain or card["power"] >= 4 or card["token"] else NEG

    def _priority(self, view, actions):
        out = []
        main, end = self._main(view), self._end(view)
        hand = view.state["self"]["hand"]
        for a in actions:
            tag = a.key[0]
            if tag == "pass":
                out.append(0.0)
                continue
            if tag == "play_land":
                out.append(80.0)
                continue
            if tag == "mana":
                out.append(NEG)  # costs expose all useful mana choices
                continue
            if tag not in {"cast", "activate"}:
                self._bad(a)
            name = a.data["card"]["name"]
            if tag == "activate":
                if name != "Lorien Revealed" or not a.data["ability"].startswith("islandcycling"):
                    self._bad(a)
                need = hand.count("Island") == 0 and self._lands(view) < PARAMETERS["desired_lands"]
                # Cycling away known junk after Brainstorm is also productive.
                junk = any(i == 0 and self._value(view, n) < 2 for i, n in view.state["self"]["library_known"])
                out.append(35.0 if (main or end) and (need or junk) else NEG)
                continue
            if tag != "cast":
                self._bad(a)
            mode, cost = a.data["mode"], ManaCost.parse(a.data["cost"]).mana_value
            allowed = {"normal", "escape"} if name == "Sleep of the Dead" else {"normal", "flashback"} if name == "Plunder the Trollshaws" else {"normal"}
            if mode not in allowed:
                self._bad(a)
            if name in {"Counterspell", "Force Spike"}:
                target = self._counter_target(view, name == "Force Spike")
                threshold = PARAMETERS["spike_min_value" if name == "Force Spike" else "counter_min_value"]
                out.append((65.0 if name == "Force Spike" else 60.0) + self._spell_value(view, target) * .1
                           if target is not None and self._spell_value(view, target) >= threshold else NEG)
                continue
            if view.context["stack"]:
                out.append(NEG)
                continue
            if name in THREATS:
                score = 45.0 if cost <= PARAMETERS["cheap_threat_cost"] else 25.0
                # Rebuild pressure on an empty board instead of saving every UU.
                if self._reserve(view, cost) and creatures(view, "self"):
                    score = NEG
                out.append(score if main else NEG)
            elif name == "Delver of Secrets":
                out.append(42.0 if main and not self._reserve(view, cost) else NEG)
            elif name in {"Deem Inferior", "Sleep of the Dead"}:
                scores = []
                for c in view.state["battlefield"]:
                    if c["controller"] != "opponent" or "Land" in c["types"]:
                        continue
                    tax = view.cards[c["name"]].get("ward", 0)
                    if available(view) < cost + tax:
                        continue
                    scores.append(self._tempo(view, c, name == "Sleep of the Dead", mode == "escape"))
                out.append(max(scores, default=NEG) if main else NEG)
            elif name == "Lorien Revealed":
                out.append(20.0 if main and view.state["self"]["library_count"] >= 3 and not self._reserve(view, cost) else NEG)
            elif name == "Plunder the Trollshaws":
                if mode not in {"normal", "flashback"}:
                    self._bad(a)
                enough = view.state["self"]["library_count"] >= (2 if mode == "flashback" else 1)
                out.append(21.0 if enough and (end or (main and not self._reserve(view, cost))) else NEG)
            elif name in CANTRIPS:
                minimum = 3 if name in {"Brainstorm", "Mental Note"} else 1
                if view.state["self"]["library_count"] < minimum:
                    out.append(NEG)
                    continue
                if name in {"Thought Scour", "Mental Note"} and self._mill_loss(view) >= PARAMETERS["known_top_keep_value"]:
                    out.append(NEG)
                    continue
                score = 28.0 if name == "Ponder" else 26.0
                if name in {"Thought Scour", "Mental Note"} and any(n in THREATS for n in hand):
                    score += 4.0
                if name == "Brainstorm" and self._lands(view) >= 2 and len(hand) <= 2:
                    score -= 8.0  # avoid locking a small hand without a shuffle
                out.append(score if end or (main and not self._reserve(view, cost)) else NEG)
            else:
                self._bad(a)
        return out

    def _target(self, view, actions):
        # Casting/activation and resolving effects remain on the public stack.
        spell = view.context["stack"][-1]["name"] if view.context["stack"] else ""
        out = []
        for a in actions:
            if a.key[0] != "target":
                self._bad(a)
            t = a.data["target"]
            if t is None:
                out.append(NEG)
            elif "sid" in t:
                item = self._stack(view, t["sid"])
                out.append(self._spell_value(view, item) if item is not None and
                           (spell != "Force Spike" or self._spike_live(view, item)) else NEG)
            elif "player" in t:
                if spell == "Thought Scour":
                    side = "opponent" if view.state["self"]["library_count"] < 3 else "self"
                    out.append(10.0 if t["player"] == side else NEG)
                else:  # public dungeon traps target the opponent
                    out.append(10.0 if t["player"] == "opponent" else NEG)
            else:
                c = permanent(view, t["oid"])
                if c is None:
                    out.append(NEG)
                elif spell in {"Deem Inferior", "Sleep of the Dead"}:
                    tax = view.cards[c["name"]].get("ward", 0)
                    cost = next((s for s in reversed(view.context["stack"]) if s["kind"] == "spell"), None)
                    # At target selection the spell's mana is not paid yet.
                    base = 0
                    if cost is not None:
                        if spell == "Deem Inferior":
                            base = max(0, 3 - view.state["self"]["cards_drawn_this_turn"]) + 1
                        else:
                            base = 3 if cost["method"] == "escape" else 1
                    out.append(self._tempo(view, c, spell == "Sleep of the Dead", cost and cost["method"] == "escape")
                               if available(view) >= base + tax else NEG)
                else:  # Initiative Forge: put counters on our best creature.
                    out.append(creature_value(c) if c["controller"] == "self" and "Creature" in c["types"] else NEG)
        return out

    def _mulligan(self, view, actions):
        hand = view.state["self"]["hand"]
        lands = hand.count("Island")
        size = 7 - view.state["self"]["mulligans"]
        supported = lands == 1 and sum(n in CANTRIPS for n in hand) >= PARAMETERS["one_land_cantrips"]
        keep = size <= 4 or 2 <= lands <= 4 or supported
        out = []
        for a in actions:
            if a.key[0] != "mulligan" or a.key[1] not in {"keep", "mulligan"}:
                self._bad(a)
            out.append(float((a.key[1] == "keep") == keep))
        return out

    def _choose_card(self, view, actions):
        out = []
        for a in actions:
            tag, name = a.key[0], (a.data.get("card") or {}).get("name")
            if tag == "search":
                out.append(10.0 if name == "Island" else NEG)
            elif tag in {"bottom", "put back", "discard"}:
                out.append(-self._value(view, name))
            elif tag in {"put", "return_gy"}:
                out.append(self._value(view, name) if name is not None else 0.0)
            else:
                self._bad(a)
        return out

    def _pay_mana(self, view, actions):
        out = []
        payment = view.context.get("payment", {})
        for a in actions:
            if a.key[0] != "pay":
                self._bad(a)
            data = a.data
            if data["via"] == "pool":
                out.append(100.0)
                continue
            if data["via"] not in {"source", "filter"}:
                self._bad(a)
            c = permanent(view, data["source"]["oid"])
            rules = view.cards[c["name"]]
            sacrifice = any(ab["sac_self"] and ab["mana"] for ab in rules["abilities"])
            flexibility = max((len(ab["mana"]) for ab in rules["abilities"] if ab["mana"]), default=1)
            out.append(-10 * sacrifice - flexibility + (1 if payment.get("colored", {}).get(data["color"], 0) else 0)
                       - (1 if data["color"] == "U" and not payment.get("colored", {}).get("U", 0) else 0))
        return out

    def _yes_no(self, view, actions):
        out = []
        known = dict(view.state["self"]["library_known"])
        for a in actions:
            tag, answer = a.key
            allowed = {"keep", "graveyard"} if tag == "explore" else {"yes", "no"}
            if answer not in allowed:
                self._bad(a)
            if tag == "reveal":
                name = known.get(0)
                want = name is not None and bool(set(view.cards[name]["types"]) & {"Instant", "Sorcery"})
                out.append(float((answer == "yes") == want))
            elif tag == "shuffle":
                name = known.get(0)
                want = name is None or self._value(view, name) < 2.0
                # A mana-starved Ponder should search again if its three cards
                # contain no Island, even when a good threat is present.
                if self._lands(view) < 2 and "Island" not in view.state["self"]["hand"]:
                    want = "Island" not in known.values()
                out.append(float((answer == "yes") == want))
            elif tag == "pay_optional":
                stack = view.context["stack"]
                top = stack[-1] if stack else None
                sid = top.get("ward", {}).get("sid") if top else None
                if sid is None and top:
                    sid = next((t["sid"] for t in top["targets"] if "sid" in t), None)
                threatened = self._stack(view, sid)
                want = threatened is not None and threatened["controller"] == "self" and (not threatened["targets"] or any(
                    "oid" not in t or permanent(view, t["oid"]) is not None for t in threatened["targets"]))
                out.append(float((answer == "yes") == want))
            elif tag == "explore":
                keep = self._value(view, known.get(0)) >= 2.0
                out.append(float((answer == "keep") == keep))
            else:
                self._bad(a)
        return out

    def _order(self, view, actions):
        out = []
        for a in actions:
            if a.key[0] not in {"order", "scry"}:
                self._bad(a)
            top, bottom = a.data["top"], a.data["bottom"]
            out.append(sum(self._value(view, n) / (i + 1) for i, n in enumerate(top)) +
                       sum(2 - self._value(view, n) for n in bottom))
        return out

    def _choose_mode(self, view, actions):
        out = []
        for a in actions:
            tag, answer = a.key
            if tag == "deem":
                if answer not in {"bottom", "second"}:
                    self._bad(a)
                # The deciding player owns the bounced card. Recover threats
                # cheaply rather than burying them; bottom unwanted mana.
                stack = view.context["stack"]
                oid = next((t["oid"] for t in stack[-1]["targets"] if "oid" in t), None) if stack else None
                c = permanent(view, oid)
                want_second = c is not None and ("Creature" in c["types"] or self._value(view, c["name"]) >= 3)
                out.append(float((answer == "second") == want_second))
            elif tag == "scry":
                if answer not in {"top", "bottom"}:
                    self._bad(a)
                known = dict(view.state["self"]["library_known"])
                keep = self._value(view, known.get(0)) >= 2
                out.append(float((answer == "top") == keep))
            elif tag == "venture":  # stable, explicit generic room preference
                out.append({"Forge": 4, "Throne of the Dead Three": 5, "Archives": 3, "Trap!": 2}.get(answer, 1))
            else:
                self._bad(a)
        return out

    def _sacrifice(self, view, actions):
        out = []
        for a in actions:
            if a.key[0] != "sacrifice":
                self._bad(a)
            c = permanent(view, a.data["card"]["oid"])
            out.append(-creature_value(c) if "Creature" in c["types"] else -self._value(view, c["name"]))
        return out

    def _exile_from_graveyard(self, view, actions):
        out = []
        for a in actions:
            if a.key[0] != "exile_gy":
                self._bad(a)
            name = a.data["card"]["name"]
            types = view.cards[name]["types"]
            fuel = bool(set(types) & {"Instant", "Sorcery"})
            recastable = name in {"Plunder the Trollshaws", "Sleep of the Dead"}
            out.append(4.0 * (not fuel) - 2.0 * recastable - self._value(view, name) * .1)
        return out

    def _choose_x(self, view, actions):
        # No X spells in the frozen list; generic positive-effect X uses max.
        for a in actions:
            if a.key[0] != "x":
                self._bad(a)
        return [a.data["amount"] for a in actions]

    def _order_triggers(self, view, actions):
        for a in actions:
            if a.key[0] != "trigger":
                self._bad(a)
        return [0.0] * len(actions)  # stable public engine order

    def _declare_attacker(self, view, actions):
        mine, theirs = creatures(view, "self"), creatures(view, "opponent")
        chosen = [c for c in mine if c["attacking"]]
        remaining = [c for c in mine if not c["attacking"] and not c["tapped"] and not c["sick"] and c["power"] > 0]
        enemy_blocks = [c for c in theirs if not c["tapped"]]
        ready = chosen + remaining
        life = view.state["opponent"]["life"] + sum(max(0, c["power"]) for c in enemy_blocks if keyword(c, "lifelink"))
        lethal_attack = damage_floor(ready, enemy_blocks) >= life
        scores = []
        for a in actions:
            if a.key[0] != "attack":
                self._bad(a)
            ref = a.data["attacker"]
            if ref is None:
                scores.append(0.0)
                continue
            c = permanent(view, ref["oid"])
            if c["power"] <= 0:
                scores.append(NEG)
                continue
            if lethal_attack:
                scores.append(100.0 + c["power"])
                continue
            holding = [b for b in mine if not b["tapped"] and not b["attacking"] and b["oid"] != c["oid"]]
            future = [e for e in theirs if e["skip_untap"] == 0]
            if damage_floor(future, holding) >= view.state["self"]["life"]:
                scores.append(NEG)
                continue
            enemies = [b for b in enemy_blocks if blocks(b, c)]
            best_trade = min((creature_value(b) for b in enemies
                              if (b["power"] >= lethal(b, c) or keyword(b, "deathtouch")) and c["power"] < lethal(c, b)), default=None)
            if best_trade is not None:
                scores.append(-creature_value(c) + best_trade)
            else:
                # Multiple small bodies can kill a ground threat. Attack when
                # their lost material pays for ours, or other pressure benefits.
                team_power = sum(max(0, b["power"]) for b in enemies)
                if enemies and team_power >= lethal(enemies[0], c):
                    scores.append(c["power"] * .1 if len(ready) > 1 else -.1)
                else:
                    scores.append(1.0 + c["power"] * .1)
        return scores

    def _declare_blocker(self, view, actions):
        blocker = permanent(view, view.context["subject"]["oid"])
        attackers = [c for c in creatures(view, "opponent") if c["attacking"]]
        blocked = {c["blocking"] for c in creatures(view, "self") if c["blocking"] is not None}
        incoming = sum(max(0, c["power"]) if c["oid"] not in blocked else
                       max(0, c["power"] - sum(lethal(c, b) for b in creatures(view, "self") if b["blocking"] == c["oid"]))
                       if keyword(c, "trample") else 0 for c in attackers)
        scores = []
        for a in actions:
            if a.key[0] != "block":
                self._bad(a)
            ref = a.data["attacker"]
            if ref is None:
                scores.append(0.0)
                continue
            attacker = permanent(view, ref["oid"])
            others = [b for b in creatures(view, "self") if b["blocking"] == attacker["oid"]]
            total = blocker["power"] + sum(b["power"] for b in others)
            kill = total >= lethal(blocker, attacker) or any(keyword(b, "deathtouch") and b["power"] > 0 for b in others)
            dies = attacker["power"] >= lethal(attacker, blocker)
            prevent = max(0, attacker["power"]) if attacker["oid"] not in blocked else 0
            if keyword(attacker, "trample"):
                prevent = min(prevent, lethal(attacker, blocker))
            score = (creature_value(attacker) if kill else 0) - (creature_value(blocker) if dies else 0)
            if not dies:
                score += prevent * .5
            if incoming >= view.state["self"]["life"] and prevent:
                score += 100 + prevent
            # Start a multi-block that the remaining legal blockers can finish.
            future = [b for b in creatures(view, "self") if not b["tapped"] and b["blocking"] is None and
                      b["oid"] > blocker["oid"] and blocks(b, attacker)]
            if not kill and total + sum(max(0, b["power"]) for b in future) >= lethal(blocker, attacker):
                score += creature_value(attacker) * .5
            scores.append(score)
        return scores

    def _damage_value(self, view, source, blockers_, split):
        score = sum(10 + creature_value(b) for b, n in zip(blockers_, split) if n >= lethal(source, b))
        if len(split) > len(blockers_):
            n = split[-1]
            score += n + (100000 if n >= view.state["opponent"]["life"] else 0)
        return score

    def _assign_damage(self, view, actions):
        out = []
        for a in actions:
            if a.key[0] != "damage":
                self._bad(a)
            source = permanent(view, a.data["source"]["oid"])
            bs = [permanent(view, b["oid"]) for b in a.data["blockers"]]
            out.append(self._damage_value(view, source, bs, a.data["split"]))
        return out

    def _assign_damage_amount(self, view, actions):
        allocation = view.context["damage"]
        bs = [permanent(view, oid) for oid in allocation["blockers"]]
        weights = [10 + creature_value(b) for b in bs]
        start = 0 if allocation["recipient"] == -1 else allocation["recipient"] + 1
        dp = _suffix_scores(weights, allocation["lethal"], allocation["remaining"], start, allocation["player_damage"] > 0)
        out = []
        for a in actions:
            if a.key[0] != "damage_amount":
                self._bad(a)
            n = a.data["amount"]
            if allocation["recipient"] == -1:
                out.append(sum(weights) + n + (100000 if n >= view.state["opponent"]["life"] else 0) if n > 0 else dp[allocation["remaining"]])
            else:
                i = allocation["recipient"]
                out.append((weights[i] if n >= allocation["lethal"][i] else 0) + dp[allocation["remaining"] - n])
        return out
