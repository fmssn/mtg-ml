"""Version 1 preboard Jund rules over detached benchmark observations.

No engine object, hidden completion, prompt, display label, random generator or
rollout is available to strategy. A fresh Position makes every choice valid
also when an episode starts in the middle of a spell's targets or costs.
"""

from collections import Counter
from collections.abc import Mapping
from functools import lru_cache

from ...engine.mana import ManaCost, RemainingCost, can_pay, pool_units
from ..adapters import AgentMetadata
from ..artifacts import FAIR
from ..views import KINDS, freeze, thaw
from . import combat
from .parameters import PARAMETERS, PARAMETERS_SHA256

T = PARAMETERS["tactics"]
P = PARAMETERS["priority"]
S = PARAMETERS["sacrifice"]
C = PARAMETERS["combat"]

NEG = -1e12

BASICS = {"Swamp": "B", "Mountain": "R", "Forest": "G"}
DRAW = {"Fanatical Offering", "Eviscerator's Insight"}


@lru_cache(maxsize=128)
def cost(text):
    return ManaCost.parse(text)


def register_jund(registry, source_revision):
    """Opt-in registration; the caller supplies the actual tested source SHA."""
    registry.register(AgentMetadata("benchmark-jund@1", "jund_wildfire", source_revision, 1,
                                    PARAMETERS_SHA256, FAIR), BenchmarkJundBot, thaw(PARAMETERS))


class UnsupportedDecision(ValueError):
    pass


class Position:
    def __init__(self, view):
        self.view, self.s, self.rules = view, view.state, view.cards
        self.me, self.opp = self.s["self"], self.s["opponent"]
        self.board = self.s["battlefield"]
        self.by_id = {c["oid"]: c for c in self.board}
        self.own = [c for c in self.board if c["controller"] == "self"]
        self.enemy = [c for c in self.board if c["controller"] == "opponent"]
        self.creatures = [c for c in self.own if c["power"] is not None]
        self.threats = [c for c in self.enemy if c["power"] is not None]
        self.lands = [c for c in self.own if "Land" in c["types"]]
        self.hand = self.me["hand"]
        self.stack = view.context["stack"]
        self.top = self.stack[-1] if self.stack else None
        self.source = self.top["source"]["name"] if self.top and self.top["source"] else None
        self.main = self.s["active"] == "self" and self.s["step"] in {"main1", "main2"} and not self.stack
        self.end = self.s["active"] == "opponent" and self.s["step"] == "end" and not self.stack
        # Priority in combat_damage is after the turn-based damage action.
        self.before_damage = self.s["step"] in {"declare_attackers", "declare_blockers"}

    def mana(self, gone=(), pool=None):
        units = pool_units(dict(self.me["pool"] if pool is None else pool))
        for c in self.own:
            if c["oid"] in gone:
                continue
            for ab in self.rules[c["name"]]["abilities"]:
                if ab["mana"] and ab["zone"] == "battlefield" and not (ab["tap"] and (c["tapped"] or c["power"] is not None and c["sick"])):
                    units.append(tuple(ab["mana"]))
                    break
        return units

    def payable(self, payment, gone=(), pool=None):
        return can_pay(RemainingCost.of(payment), self.mana(gone, pool))

    def spell_cost(self, name, mode="normal", x=0, extra_artifacts=0):
        r = self.rules[name]
        base = r[mode] if mode in {"flashback", "bestow"} else r["cost"]
        c = cost(base).with_x(x)
        if name == "Refurbished Familiar":
            c = c.reduced(extra_artifacts + sum("Artifact" in p["types"] for p in self.own))
        return c

    def remaining_basics(self):
        used = Counter(self.hand) + Counter(self.me["graveyard"]) + Counter(self.me["exile"])
        used.update(c["name"] for c in self.own)
        used.update(it["source"]["name"] for it in self.stack if it["kind"] == "spell" and it["controller"] == "self")
        return {n: max(0, self.view.own_deck.get(n, 0) - used[n]) for n in BASICS}

    def land_need(self):
        return PARAMETERS["land_goal"] - len(self.lands) - sum("Land" in self.rules[n]["types"] for n in self.hand)

    def card_value(self, name):
        if "Land" in self.rules[name]["types"]:
            return 4. if self.land_need() > 0 else 1.
        v = PARAMETERS["cards"].get(name, 1.)
        if name == "Cleansing Wildfire" and not any("indestructible" in c["keywords"] for c in self.lands):
            v -= 1.5
        if name == "Toxin Analysis" and any(c["name"] == "Krark-Clan Shaman" for c in self.creatures):
            v += T["engine_threat"]
        return v

    def growth(self, fodder):
        return sum(T["growth"] for c in self.creatures if c["oid"] != fodder["oid"] and
                   (c["name"] == "Gixian Infiltrator" or c["name"] == "Writhing Chrysalis" and "Eldrazi" in self.rules[fodder["name"]]["subtypes"]))

    def sacrifice_cost(self, c):
        n = c["name"]
        if "Land" in c["types"]:
            v = S["needed_land"] if len(self.lands) < PARAMETERS["land_goal"] else S["extra_land"]
        elif n == "Ichor Wellspring":
            v = S["wellspring"]
        elif n == "Nihil Spellbomb":
            v = S["cantrip"] if self.payable(cost("{B}")) else S["spellbomb_unpaid"]
        elif n in {"Clue", "Map", "Lembas"}:
            v = S["cantrip"]
        elif c["token"]:
            v = S["token"] + combat.value(c)
        elif c["power"] is not None:
            v = S["creature"] + combat.value(c)
        else:
            v = S["other"]
        # A threatened permanent is cheap to cash in; a required blocker isn't.
        if any(it["controller"] == "opponent" and ("perm", c["oid"]) in it["targets"] for it in self.stack):
            v = min(v, S["threatened"])
        return v - self.growth(c)

    def fodder(self, artifact=False, payment=None):
        out = []
        for c in self.own:
            if "Artifact" not in c["types"] and (artifact or c["power"] is None):
                continue
            self_sac = any(ab["mana"] and ab["sac_self"] for ab in self.rules[c["name"]]["abilities"])
            if payment is not None and not self.payable(payment, (c["oid"],) if self_sac else ()):
                continue
            out.append(c)
        return sorted(out, key=lambda c: (self.sacrifice_cost(c), c["oid"]))

    def munitions_budget(self):
        """Maximum one-mana lethal shots with all fodder, without double-use.

        A Spawn used as fodder removes one potential mana unit. Other
        artifacts may tap for mana before being sacrificed. For k shots the
        necessary Spawn fodder is max(0, k - other), so k + that <= mana.
        """
        fodder = self.fodder(payment=cost("{1}"))
        other = sum(c["name"] != "Eldrazi Spawn" for c in fodder)
        mana = len(self.mana())
        return min(len(fodder), mana, (mana + other) // 2)

    def pending_damage(self, oid):
        return sum(1 for it in self.stack if it["source"] and it["source"]["name"] == "Makeshift Munitions"
                   and it["controller"] == "self" and ("perm", oid) in it["targets"])

    def removal_value(self, c, payment=None):
        if c["controller"] != "opponent" or "indestructible" in c["keywords"]:
            return NEG
        if any(it["controller"] == "self" and it["source"] and it["source"]["name"] == "Cast Down"
               and ("perm", c["oid"]) in it["targets"] for it in self.stack):
            return NEG
        ward = self.rules[c["name"]]["ward"]
        if not self.payable((payment or ManaCost()).plus(ManaCost(ward))):
            return NEG
        v = combat.value(c)
        if c["attacking"] and self.before_damage:
            v += T["ward_attack"]
            if sum(max(0, a["power"]) for a in self.threats if a["attacking"]) >= self.me["life"]:
                v += C["survival"]
        if c["name"] in {"Krark-Clan Shaman", "Gixian Infiltrator"}:
            v += T["engine_threat"]
        return v

    def reserve(self):
        if "Cast Down" in self.hand:
            best = max(self.threats, key=lambda c: self.removal_value(c, cost("{1}{B}")), default=None)
            if best is not None and self.removal_value(best, cost("{1}{B}")) >= PARAMETERS["removal_min"]:
                return cost("{1}{B}").plus(ManaCost(self.rules[best["name"]]["ward"]))
        return ManaCost()

    def sweep(self, shaman, toxin=False):
        """Best additional activation count and its net gain over pending damage."""
        pending = sum(it["kind"] == "ability" and it["source"] and it["source"]["oid"] == shaman["oid"]
                      for it in self.stack)
        fodder = self.fodder(artifact=True)
        dt = toxin or "deathtouch" in shaman["keywords"]
        link = toxin or "lifelink" in shaman["keywords"]

        def outcome(n, buffed=True):
            gone = {c["oid"] for c in fodder[:n]}
            creatures = [dict(c) for c in self.board if c["power"] is not None and c["oid"] not in gone]
            for c in creatures:
                if c["controller"] == "self" and c["name"] == "Gixian Infiltrator":
                    growth = n + sum(it["kind"] == "trigger" and it["source"] and it["source"]["oid"] == c["oid"] for it in self.stack)
                    c["power"] += growth
                    c["toughness"] += growth
            total = -sum(self.sacrifice_cost(c) for c in fodder[:n])
            gain = 0
            for _ in range(n + pending):
                survivors = []
                for c in creatures:
                    if "flying" in c["keywords"]:
                        survivors.append(c)
                        continue
                    gain += int(link if buffed else "lifelink" in shaman["keywords"])
                    if combat.dies(c, 1, dt if buffed else "deathtouch" in shaman["keywords"]):
                        total += combat.value(c) * (1 if c["controller"] == "opponent" else -1)
                    else:
                        c["damage"] += 1
                        survivors.append(c)
                creatures = survivors
            total += gain * (T["gain_low"] if self.me["life"] < T["low_life"] else T["gain_normal"])
            if self.before_damage and any(c["attacking"] for c in self.threats):
                before = sum(max(0, c["power"]) for c in self.threats if c["attacking"])
                after = sum(max(0, c["power"]) for c in creatures if c["controller"] == "opponent" and c["attacking"])
                if before >= self.me["life"] and after < self.me["life"] + gain:
                    total += C["survival"]
            return total

        base = outcome(0, buffed=False)
        best, count = 0., 0
        for n in range(0 if toxin else 1, len(fodder) + 1):
            score = outcome(n) - base
            if score > best:
                best, count = score, n
        return best, count

    def toxin_value(self, c):
        if c["controller"] != "self" or "deathtouch" in c["keywords"]:
            return NEG
        if c["name"] == "Krark-Clan Shaman":
            return self.sweep(c, toxin=True)[0] - T["toxin_cost"]
        score = 0.
        attackers = [t for t in self.threats if t["attacking"]]
        if attackers and self.s["step"] == "declare_attackers" and not c["tapped"]:
            buffed = [dict(b, keywords=tuple(set(b["keywords"]) | {"deathtouch", "lifelink"})) if b["oid"] == c["oid"] else b for b in self.creatures]
            def loss(blockers):
                plan = combat.block_plan(attackers, blockers, self.me["life"])
                results = [combat.exchange(a, plan[a["oid"]]) for a in attackers]
                return sum(r[0] - r[3] for r in results)
            if loss(self.creatures) >= self.me["life"] > loss(buffed):
                score += C["survival"]
        for other in self.threats:
            if c["blocking"] == other["oid"] or other["blocking"] == c["oid"]:
                if c["power"] > 0 and not combat.dies(other, c["power"]):
                    score += combat.value(other)
        if c["attacking"] or c["blocking"] is not None:
            score += max(0, c["power"]) * (T["gain_low"] if self.me["life"] < T["low_life"] else T["gain_normal"])
        return score - T["toxin_cost"]

    def mana_pressure(self, gone=()):
        units = self.mana(gone)
        need = Counter()
        for name in self.hand:
            for col, n in cost(self.rules[name]["cost"]).colored:
                need[col] += n * (2 if "Instant" in self.rules[name]["types"] else 1)
        return {col: n / max(1, sum(col in u for u in units)) for col, n in need.items()}

    def known_counter(self):
        """Evidence only: known card plus enough untapped public blue sources."""
        units = pool_units(dict(self.opp["pool"]))
        for c in self.enemy:
            if not c["tapped"]:
                units.extend(tuple(ab["mana"]) for ab in self.rules[c["name"]]["abilities"] if ab["mana"] and ab["tap"])
        return "Counterspell" in self.opp["hand_known"] and can_pay(RemainingCost.of(cost("{U}{U}")), units)

    def retained_value(self, names):
        """Bottom against the retained hand's land count, colors and early plays."""
        lands = [n for n in names if "Land" in self.rules[n]["types"]]
        units = [tuple(ab["mana"]) for n in lands for ab in self.rules[n]["abilities"] if ab["mana"]]
        if "Twisted Landscape" in lands and any(self.remaining_basics().values()):
            units.remove(("C",))
            units.append(tuple(BASICS[n] for n, count in self.remaining_basics().items() if count))
        spells = [n for n in names if n not in lands]
        early = sum(can_pay(RemainingCost.of(self.spell_cost(n, x=int(n == "Nyxborn Hydra"))), units) for n in spells)
        colors = {color for n in spells for color, _ in cost(self.rules[n]["cost"]).colored}
        missing = sum(not any(color in unit for unit in units) for color in colors)
        return sum(self.card_value(n) for n in spells) + T["early_play"] * early - T["color_access"] * missing - P["land"] * max(0, 2 - len(lands)) - max(0, len(lands) - 3)

    def fetch_value(self, name):
        color = BASICS.get(name)
        if color is None:
            return NEG if name is None else self.card_value(name)
        supply = Counter(col for c in self.lands for a in self.rules[c["name"]]["abilities"] for col in (a["mana"] or ()))
        need = Counter(col for n in self.hand for col, count in cost(self.rules[n]["cost"]).colored for _ in range(count))
        return 5 * (supply[color] == 0) + need[color] / (supply[color] + 1)


class BenchmarkJundBot:
    name = "benchmark-jund@1"
    handlers = frozenset(KINDS)

    def __init__(self):
        self.own_deck = None

    def reset(self, own_deck: Mapping[str, int]):
        self.own_deck = freeze(dict(own_deck))

    def observe(self, visible_event):
        # Current permitted state carries every fact v1 uses. No game history or
        # opponent private menu is retained; mid-line resets behave identically.
        pass

    def choose(self, player_view, legal_actions):
        if self.own_deck is None:
            raise ValueError("benchmark-jund@1: reset required")
        if not legal_actions:
            raise UnsupportedDecision("empty action sequence")
        kind = legal_actions[0].kind
        handler = getattr(self, "choose_" + kind, None)
        if handler is None or any(a.kind != kind or a.index != i for i, a in enumerate(legal_actions)):
            raise UnsupportedDecision(f"benchmark-jund@1: unsupported or malformed {kind}")
        p = Position(player_view)
        scores = handler(p, legal_actions)
        return max(range(len(legal_actions)), key=lambda i: (scores[i], -i))

    def choose_priority(self, p, actions):
        return [self.priority(p, a) for a in actions]

    def priority(self, p, a):
        verb = a.key[0]
        if verb == "pass":
            return 0.
        c = a.data["card"]
        name = c["name"]
        if verb == "play_land":
            rules = p.rules[name]
            new_units = p.mana() + ([] if rules["enters_tapped"] else [tuple(next(ab["mana"] for ab in rules["abilities"] if ab["mana"]))])
            enabled = max((PARAMETERS["curve"].get(n, 5.) for n in p.hand if "Land" not in p.rules[n]["types"]
                           and can_pay(RemainingCost.of(p.spell_cost(n, x=int(n == "Nyxborn Hydra"), extra_artifacts=int("Artifact" in rules["types"]))), new_units)
                           and not p.payable(p.spell_cost(n, x=int(n == "Nyxborn Hydra")))), default=0.)
            bridge = "indestructible" in rules["keywords"]
            return P["land"] + enabled + (3 if bridge and "Cleansing Wildfire" in p.hand else 1 if bridge else 0)
        if verb == "mana":
            return self.spawn_value(p, p.by_id[c["oid"]])
        if verb == "activate":
            return self.activation(p, a)
        if verb != "cast":
            raise UnsupportedDecision(f"unsupported Jund priority action {a.key}")
        payment = cost(a.data["cost"])
        mode = a.data["mode"]
        if name == "Cast Down":
            v = max((p.removal_value(c, payment) for c in p.threats), default=NEG)
            return P["removal"] + v if v >= PARAMETERS["removal_min"] else NEG
        if name == "Toxin Analysis":
            v = max((p.toxin_value(c) for c in p.creatures), default=NEG)
            # Do not invest a combo into a source already targeted by a known
            # removal spell unless the actual combat trick itself saves it.
            return P["toxin"] + v if v > 0 else NEG
        if name in DRAW:
            fodder = p.fodder(payment=payment)
            if not fodder:
                return NEG
            loss = p.sacrifice_cost(fodder[0])
            threatened = any(it["controller"] == "opponent" and ("perm", fodder[0]["oid"]) in it["targets"] for it in p.stack)
            if threatened:
                return P["save_fodder"] - loss
            if loss > T["cheap_fodder"] or not (p.end or p.main):
                return NEG
            return P["draw"] - loss + (2 if fodder[0]["name"] == "Ichor Wellspring" else 0) - (1 if mode == "flashback" else 0)
        if name == "Cleansing Wildfire":
            bridge = any("indestructible" in c["keywords"] for c in p.lands)
            return P["wildfire"] if p.main and bridge and any(p.remaining_basics().values()) else NEG
        if not p.main:
            return NEG
        if name not in PARAMETERS["curve"]:
            raise UnsupportedDecision(f"unsupported preboard spell {name}")
        reserve = p.reserve()
        if not reserve.is_zero() and not p.payable(payment.plus(reserve)):
            return NEG
        if name == "Nyxborn Hydra":
            minimum = p.spell_cost(name, mode, x=1).plus(reserve)
            if not p.payable(minimum):
                return NEG
            if mode == "bestow":
                ready = [c for c in p.creatures if not c["sick"] and not c["tapped"]]
                return P["bestow_attack"] if ready and p.s["step"] == "main1" else P["bestow_later"]
        # When an actual Counterspell is known and castable, prefer the cheaper
        # development spell as bait. Unknown hand cards carry no inferred identity.
        risk = T["counter_risk"] * payment.mana_value if p.known_counter() else 0
        return PARAMETERS["curve"][name] + .1 * payment.mana_value - risk

    def spawn_value(self, p, spawn):
        if spawn["name"] != "Eldrazi Spawn":
            return NEG
        before_combat = p.s["active"] == "self" and p.s["step"] in {"main1", "begin_combat"}
        if not before_combat and not p.before_damage:
            return NEG
        grown = [dict(c, power=c["power"] + 1, toughness=c["toughness"] + 1) if c["name"] in {"Writhing Chrysalis", "Gixian Infiltrator"} else c
                 for c in p.creatures if c["oid"] != spawn["oid"]]
        attacks = [c for c in grown if c["attacking"] or before_combat and not c["sick"] and not c["tapped"]]
        declared = p.s["step"] == "declare_blockers"
        blocks = {c["oid"]: [b for b in p.threats if b["blocking"] == c["oid"]] for c in attacks} if declared else combat.block_plan(attacks, p.threats, p.opp["life"])
        results = [combat.exchange(c, blocks[c["oid"]], declared and c["oid"] in p.view.context["blocked"]) for c in attacks]
        damage = sum(r[0] - r[3] for r in results)
        if damage >= p.opp["life"]:
            return C["lethal"]
        for c in grown:
            if c["blocking"] is not None:
                a = p.by_id.get(c["blocking"])
                if a and c["power"] >= combat.lethal(c, a) and c["toughness"] - c["damage"] > a["power"]:
                    old = p.by_id[c["oid"]]
                    if combat.dies(old, a["power"]):
                        return C["survival"]
        return NEG

    def activation(self, p, a):
        name, ref = a.data["card"]["name"], a.data["card"]
        c = p.by_id.get(ref["oid"])
        if name == "Krark-Clan Shaman":
            # A Toxin still on the stack must resolve before using its keywords.
            if any(it["source"] and it["source"]["name"] == "Toxin Analysis" and ("perm", c["oid"]) in it["targets"] for it in p.stack):
                return NEG
            v, n = p.sweep(c)
            return P["sweep"] + v if n and v > 0 else NEG
        if name == "Makeshift Munitions":
            fodder = p.fodder(payment=cost("{1}"))
            cheap = [c for c in fodder if p.sacrifice_cost(c) <= T["cheap_fodder"]]
            budget = p.munitions_budget()
            pending = sum(it["source"] and it["source"]["name"] == name and ("player", "opponent") in it["targets"] for it in p.stack)
            if budget and 0 < p.opp["life"] - pending <= budget:
                return C["lethal"]
            targets = [t for t in p.threats if combat.dies(t, 1 + p.pending_damage(t["oid"])) and not combat.dies(t, p.pending_damage(t["oid"]))
                       and p.removal_value(t, cost("{1}")) > 0]
            return P["munitions"] + max(map(combat.value, targets)) if cheap and targets else NEG
        if name == "Nihil Spellbomb":
            fuel = sum(bool({"Instant", "Sorcery"}.intersection(p.rules[n]["types"])) for n in p.opp["graveyard"])
            return P["graveyard"] if fuel >= PARAMETERS["spellbomb_fuel"] or "Eviscerator's Insight" in p.opp["graveyard"] or "Sleep of the Dead" in p.opp["graveyard"] else NEG
        if name == "Lembas":
            incoming = sum(t["power"] for t in p.threats if t["attacking"]) if p.before_damage else 0
            return C["survival"] if incoming >= p.me["life"] else P["life"] if p.me["life"] <= PARAMETERS["lembas_life"] and p.end else NEG
        if name == "Clue":
            return P["utility"] if p.end or p.main and len(p.hand) <= 1 else NEG
        if name == "Map":
            return P["utility"] if p.main and p.creatures else NEG
        if name == "Twisted Landscape":
            if ref["zone"] == "hand":
                return P["utility"] if p.end and p.land_need() < 0 else NEG
            need = any(p.fetch_value(n) >= 5 and count for n, count in p.remaining_basics().items())
            return P["fetch"] if need and (p.main or p.end) else 2. if p.end and any(p.remaining_basics().values()) else NEG
        raise UnsupportedDecision(f"unsupported preboard activation {a.key}")

    def face_lethal(self, p):
        pending = sum(it["source"] and it["source"]["name"] == "Makeshift Munitions" and
                      ("player", "opponent") in it["targets"] for it in p.stack)
        return 0 < p.opp["life"] - pending <= p.munitions_budget()

    def choose_target(self, p, actions):
        def score(a):
            t = a.data["target"]
            if t is None:
                return 0.
            c = p.by_id.get(t.get("oid"))
            name = p.source
            if name == "Cast Down":
                # Exclude the spell currently being built from pending-removal
                # detection; its chosen target list is still empty here.
                return p.removal_value(c, p.spell_cost(name)) if c else NEG
            if name == "Cleansing Wildfire":
                return 10. if c and c["controller"] == "self" and "indestructible" in c["keywords"] else NEG
            if name == "Toxin Analysis":
                return p.toxin_value(c) if c else NEG
            if name == "Makeshift Munitions":
                if t.get("player"):
                    return C["lethal"] if t["player"] == "opponent" and self.face_lethal(p) else NEG
                return P["removal"] + combat.value(c) if c and p.removal_value(c, cost("{1}")) > 0 and combat.dies(c, 1 + p.pending_damage(c["oid"])) else NEG
            if name == "Nihil Spellbomb":
                return 1. if t.get("player") == "opponent" else NEG
            if name in {"Map", "Nyxborn Hydra"}:
                return combat.value(c) + (3 if not c["sick"] else 0) + (3 if "flying" in c["keywords"] else 0) if c and c["controller"] == "self" else NEG
            raise UnsupportedDecision(f"unsupported target source {name}")
        return [score(a) for a in actions]

    def choose_pay_mana(self, p, actions):
        rem = p.view.context["payment"]
        reserve = p.reserve() if p.source != "Cast Down" else ManaCost()
        scores = []
        for a in actions:
            data, pool = a.data, dict(p.me["pool"])
            rest = RemainingCost(rem["generic"], dict(rem["colored"]))
            rest.apply(data["color"])
            gone = []
            hurt = 0.
            if data["via"] == "pool":
                pool[data["color"]] -= 1
            else:
                c = p.by_id[data["source"]["oid"]]
                gone = [c["oid"]]
                mana = next(ab for ab in p.rules[c["name"]]["abilities"] if ab["mana"])
                need = p.mana_pressure()
                hurt = sum(need.get(col, 0) for col in mana["mana"]) + .1 * len(mana["mana"])
                if mana["sac_self"]:
                    hurt += p.sacrifice_cost(c) + 2
            remaining = ManaCost(rest.generic, tuple(sorted(rest.colored.items())))
            preserves = p.payable(remaining.plus(reserve), gone, pool)
            scores.append(100 * preserves - hurt + (.01 if data["via"] == "pool" else 0))
        return scores

    def choose_sacrifice(self, p, actions):
        return [-p.sacrifice_cost(p.by_id[a.data["card"]["oid"]]) + (.01 + .01 * p.mana_pressure().get(a.data.get("color"), 0) if a.data.get("via") == "source" else 0) for a in actions]

    def choose_choose_x(self, p, actions):
        reserve = p.reserve()
        return [a.data["amount"] if p.payable(p.spell_cost(p.source, p.top["method"], a.data["amount"]).plus(reserve)) else NEG for a in actions]

    def choose_mulligan(self, p, actions):
        size = 7 - p.me["mulligans"]
        lands = [n for n in p.hand if "Land" in p.rules[n]["types"]]
        units = [tuple(ab["mana"]) for n in lands for ab in p.rules[n]["abilities"] if ab["mana"]]
        if "Twisted Landscape" in lands:
            units += [tuple(BASICS.values())]  # future fetched basic replaces Landscape
            units.remove(("C",))
        cheap = any("Land" not in p.rules[n]["types"] and p.spell_cost(n, x=int(n == "Nyxborn Hydra")).mana_value <= 2
                    and can_pay(RemainingCost.of(p.spell_cost(n, x=int(n == "Nyxborn Hydra"))), units) for n in p.hand)
        lo, hi = PARAMETERS["keep_lands"].get(str(size), (1, 5))
        keep = size <= 4 or lo <= len(lands) <= hi and cheap
        return [float((a.key[1] == "keep") == keep) for a in actions]

    def choose_choose_card(self, p, actions):
        scores = []
        for a in actions:
            verb = a.key[0]
            c = a.data["card"]
            name = None if c is None else c["name"]
            if verb == "search":
                scores.append(p.fetch_value(name))
            elif verb == "bottom":
                retained = list(p.hand)
                retained.remove(name)
                scores.append(p.retained_value(retained))
            elif verb in {"discard", "put_back"}:
                scores.append(-p.card_value(name))
            elif verb in {"put", "dig"}:
                scores.append(0. if name is None else p.card_value(name))
            else:
                raise UnsupportedDecision(f"unsupported Jund card choice {a.key}")
        return scores

    def choose_yes_no(self, p, actions):
        out = []
        for a in actions:
            tag, ans = a.key
            if tag == "pay_optional":
                pay = p.me["library_count"] > 0 if p.source == "Nihil Spellbomb" else True
                out.append(float((ans == "yes") == pay))
            elif tag == "explore":
                known = dict(p.me["library_known"])
                keep = 0 in known and p.card_value(known[0]) >= 2
                out.append(float((ans == "keep") == keep))
            elif tag == "shuffle":
                out.append(float(ans == "no"))
            else:
                raise UnsupportedDecision(f"unsupported Jund yes/no {tag}")
        return out

    def choose_choose_mode(self, p, actions):
        out = []
        for a in actions:
            tag, ans = a.key
            if tag in {"scry", "surveil"}:
                known = dict(p.me["library_known"])
                keep = 0 in known and p.card_value(known[0]) >= 2
                out.append(float((ans == "top") == keep))
            elif tag == "deem":
                targets = p.top["targets"] if p.top else ()
                c = next((p.by_id.get(r[1]) for r in targets if r[0] == "perm"), None)
                keep = c is not None and p.card_value(c["name"]) >= 2
                out.append(float((ans == "second") == keep))
            else:
                raise UnsupportedDecision(f"unsupported Jund mode {a.key}")
        return out

    def choose_order(self, p, actions):
        return [sum((len(a.data["top"]) - i) * p.card_value(n) for i, n in enumerate(a.data["top"]))
                + sum(2 - p.card_value(n) for n in a.data["bottom"]) for a in actions]

    def choose_order_triggers(self, p, actions):
        # First placed resolves last: growth before draw keeps new information
        # available before committing any subsequent main-phase action.
        return [1. if a.key[1] in {"Gixian Infiltrator", "Writhing Chrysalis"} else 0. for a in actions]

    def choose_exile_from_graveyard(self, p, actions):
        return [0. if a.data["card"] is None else -p.card_value(a.data["card"]["name"]) for a in actions]

    def choose_declare_attacker(self, p, actions):
        # Legal options deduplicate equivalent creatures, but a combined attack
        # needs every physical copy, not just the currently offered representative.
        eligible = [c for c in p.creatures if not c["tapped"] and not c["attacking"] and
                    (not c["sick"] or "haste" in c["keywords"]) and "defender" not in c["keywords"]]
        picked = combat.attack_plan(eligible, [c for c in p.creatures if c["attacking"]], p.threats,
                                    p.creatures, p.threats, p.me["life"], p.opp["life"])
        return [0. if a.data["attacker"] is None else 1. if a.data["attacker"]["oid"] in picked else NEG for a in actions]

    def choose_declare_blocker(self, p, actions):
        attackers = [c for c in p.threats if c["attacking"]]
        fixed = {a["oid"]: [b for b in p.creatures if b["blocking"] == a["oid"]] for a in attackers}
        blocker = actions[0].data["blocker"]["oid"]
        cursor = next(i for i, c in enumerate(p.creatures) if c["oid"] == blocker)
        plan = combat.block_plan(attackers, p.creatures[cursor:], p.me["life"], fixed)
        wanted = next((aid for aid, bs in plan.items() if any(b["oid"] == blocker for b in bs)), None)
        return [1. if (None if a.data["attacker"] is None else a.data["attacker"]["oid"]) == wanted else 0. for a in actions]

    def choose_assign_damage(self, p, actions):
        scores = []
        for a in actions:
            src = p.by_id[a.data["source"]["oid"]]
            blockers = [p.by_id[b["oid"]] for b in a.data["blockers"]]
            split = a.data["split"]
            face = split[-1] if len(split) > len(blockers) else 0
            scores.append((C["lethal"] if face >= p.opp["life"] else 0) + face + sum(C["kill"] + combat.value(b) for b, n in zip(blockers, split) if combat.dies(b, n, "deathtouch" in src["keywords"])))
        return scores

    def choose_assign_damage_amount(self, p, actions):
        d = p.view.context["damage"]
        weights = [0. if "indestructible" in p.by_id[oid]["keywords"] else C["kill"] + combat.value(p.by_id[oid]) for oid in d["blockers"]]
        start = 0 if d["recipient"] == -1 else d["recipient"] + 1
        dp = combat.suffix_values(d["remaining"], d["lethal"][start:], weights[start:], d["player_damage"] > 0)
        out = []
        for a in actions:
            n = a.data["amount"]
            if d["recipient"] == -1:
                out.append((C["lethal"] if n >= p.opp["life"] else 0) + sum(weights) + n if n > 0 else dp[d["remaining"]])
            else:
                i = d["recipient"]
                out.append((weights[i] if n >= d["lethal"][i] else 0) + dp[d["remaining"] - n])
        return out
