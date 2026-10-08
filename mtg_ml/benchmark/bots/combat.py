"""Static combat arithmetic for benchmark specialists; no game copies or rollout."""

from itertools import accumulate
from collections import deque


def value(c):
    if c is None or c.get("power") is None:
        return 0.0
    if c["token"] and c["power"] == 0:
        return 0.3
    return max(0, c["power"]) * 1.5 + max(0, c["toughness"]) * .5 + 1.5 * ("flying" in c["keywords"])


def can_block(a, b):
    return not b["tapped"] and "unblockable" not in a["keywords"] and (
        "flying" not in a["keywords"] or bool({"flying", "reach"}.intersection(b["keywords"])))


def lethal(a, b):
    return 1 if "deathtouch" in a["keywords"] else max(0, b["toughness"] - b["damage"])


def dies(c, damage, deathtouch=False):
    return "indestructible" not in c["keywords"] and (damage > 0 and deathtouch or damage + c["damage"] >= c["toughness"])


def suffix_values(power, thresholds, weights, must_kill=False):
    """Best material value using exactly r damage on a blocker suffix, O(nP)."""
    dp = [0.] + [-float("inf")] * power
    for need, weight in reversed(list(zip(thresholds, weights))):
        prefix, window, nxt = list(accumulate(dp, max)), deque(), []
        for r in range(power + 1):
            kill = prefix[r - need] + weight if r >= need else -float("inf")
            while window and dp[window[-1]] <= dp[r]:
                window.pop()
            window.append(r)
            while window and window[0] <= r - need:
                window.popleft()
            spare = dp[window[0]] if window and not must_kill else -float("inf")
            nxt.append(max(kill, spare))
        dp = nxt
    return dp


def exchange(a, blockers):
    """(damage to defender, attacker material gain, attacker lifegain).

    Assign damage to maximize kills after reserving maximum legal trample.
    Combat in the frozen lists has no first/double strike.
    """
    power = max(0, a["power"])
    if not blockers:
        return power, 0., power if "lifelink" in a["keywords"] else 0
    thresholds = [lethal(a, b) for b in blockers]
    face = max(0, power - sum(thresholds)) if "trample" in a["keywords"] else 0
    weights = [0. if "indestructible" in b["keywords"] else value(b) for b in blockers]
    material = suffix_values(power - face, thresholds, weights)[power - face]
    if dies(a, sum(max(0, b["power"]) for b in blockers), any("deathtouch" in b["keywords"] and b["power"] > 0 for b in blockers)):
        material -= value(a)
    return face, material, power if "lifelink" in a["keywords"] else 0


def block_plan(attackers, blockers, life, fixed=None):
    """Deterministic group heuristic: single blocks and cheapest lethal gangs.

    This optimizes a static assignment, never branches through game actions.
    Existing blocks are fixed. Recompute after each actual block decision.
    """
    plan = {a["oid"]: list((fixed or {}).get(a["oid"], ())) for a in attackers}
    used = {b["oid"] for group in plan.values() for b in group}
    free = [b for b in blockers if b["oid"] not in used and not b["tapped"]]

    def score(p):
        results = [exchange(a, p[a["oid"]]) for a in attackers]
        damage = sum(r[0] for r in results)
        # Avoid a lethal attack first; otherwise preserve material over chumps.
        return (-10000 if damage >= life else 0) - sum(r[1] for r in results) - .35 * damage

    while free:
        best, chosen = score(plan), None
        for a in attackers:
            eligible = [b for b in free if can_block(a, b)]
            eligible.sort(key=lambda b: (value(b), b["oid"]))
            groups = [[b] for b in eligible]
            gang = []
            for b in eligible:
                gang.append(b)
                if len(gang) > 1:
                    groups.append(list(gang))
            for group in groups:
                candidate = dict(plan)
                candidate[a["oid"]] = plan[a["oid"]] + group
                s = score(candidate)
                if s > best + 1e-9:
                    best, chosen = s, (a["oid"], group)
        if chosen is None:
            break
        aid, group = chosen
        plan[aid] += group
        ids = {b["oid"] for b in group}
        free = [b for b in free if b["oid"] not in ids]
    return plan


def attack_plan(eligible, committed, defenders, reserves, enemies, own_life, enemy_life):
    """Compare all-in, individual additions and all-in-minus-one attack groups."""
    groups = [[], eligible] + [[a] for a in eligible] + [eligible[:i] + eligible[i + 1:] for i in range(len(eligible))]
    best, picked = -float("inf"), []
    for group in groups:
        attack = committed + group
        blocks = block_plan(attack, defenders, enemy_life)
        result = [exchange(a, blocks[a["oid"]]) for a in attack]
        damage, material, gain = (sum(r[i] for r in result) for i in range(3))
        tapped = {a["oid"] for a in attack if "vigilance" not in a["keywords"]}
        left = [c for c in reserves if c["oid"] not in tapped and not c["tapped"]]
        # Do not credit opponents killed by a predicted block as certain losses.
        crack = [dict(c, tapped=False) for c in enemies]
        defense = block_plan(crack, left, own_life + gain)
        incoming = sum(exchange(a, defense[a["oid"]])[0] for a in crack)
        score = (100000 if damage >= enemy_life else 0) - (10000 if incoming >= own_life + gain else 0)
        score += material + damage * .7 + gain * .4 - incoming * .15
        if score > best + 1e-9:
            best, picked = score, group
    return {c["oid"] for c in picked}
