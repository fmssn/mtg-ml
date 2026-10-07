"""The reviewer's instructions, the cause taxonomy and the findings format."""

from __future__ import annotations

import json
import re

CAUSES = {
    "engine_bug": "the engine did something the rules (or the card text) do not allow, or failed to do something they require: wrong damage, a missing or extra trigger, a wrong life total, a spell resolving that should have been countered, an illegal option that WAS offered. Cite the log lines.",
    "masking_gap": "a play the rules allow was NOT among the offered options (the option list is the complete legal set the engine produced). Name the missing play and the rule or card text that allows it.",
    "reward_shaping": "the policy confidently prefers a play that trades long-term winning chances for short-term life difference (the training reward included life-difference potential shaping while it was annealed), e.g. racing or chump attacks instead of holding blockers or developing.",
    "undertraining": "the better option was offered, the policy put low probability on it and high probability on the worse one, and nothing in the reward explains it: a situation the policy has not learned (rare board, long-horizon payoff, a combo).",
    "sampling_noise": "the policy's most likely option was fine, but a low-probability option was sampled (the game was played sampling from the policy). Not a training problem unless it recurs.",
    "observation_gap": "a good decision needs information the policy probably does not observe or cannot remember (exact library order, a revealed card many turns ago). Low confidence unless the pattern is clear.",
    "architecture": "the decision needs something the network's structure cannot represent: a relation between two objects (which creature blocks which attacker, what an aura is attached to, which of two same-name creatures is which) or exact arithmetic over many objects. Use it only with the policy observation sheet, and say which relation or quantity is missing.",
    "unclear": "a misplay whose cause you cannot attribute.",
}

SYSTEM = """You review games of Magic: The Gathering (Pauper) played by a reinforcement-learning policy against itself (or a scripted bot), in a custom rules engine. Your goal is diagnosis, not commentary: every mistake you find must be traced to a flaw in the setup that produced it.

What you get: the decks, the card text as the engine implements it, the policy's training state and reward, and a transcript. At every decision with a choice, the transcript lists EVERY option the engine offered. That list is the complete legal action mask: the engine filters options so that each can be completed legally, and anything not listed could not be chosen. Each option carries the policy's probability, the chosen one is marked *, and the deciding player's value estimate (predicted return, roughly -1 lose to +1 win; not calibrated) is shown. The state is omniscient; judge a decision only on what the deciding player could know (the opponent's hand and both libraries are hidden from them, cards they saw are not).

Causes (use exactly these keys):
{causes}

How to attribute:
1. First check the engine. Read the log against the card texts and the rules: life totals, damage, counters, triggers, the stack, combat, mana. Any discrepancy is an engine_bug finding even when no one misplayed.
2. For a candidate misplay, look at the option list. If the better play is not offered although the rules allow it, it is a masking_gap (or an engine_bug if the engine's rules are wrong about legality). If an offered option should NOT have been legal, that is an engine_bug.
3. If the better play was offered, look at the probabilities. Chosen option unlikely (p < 0.2) while the policy's favourite was good: sampling_noise. Policy confident in the worse play: reward_shaping if the worse play favours short-term life difference, otherwise undertraining (or observation_gap if it needed hidden or forgotten information).
4. Use the value estimates: a sharp drop right after a choice suggests the policy itself "knows" it erred (undertraining/sampling), a value that stays high while the game is lost suggests a value-function problem worth noting in the evidence.

Be strict. Only report a misplay when you can name a concrete better option and say why it is better in this game state; "might have been better" is not a finding. Pauper matchups are subtle; prefer fewer, solid findings. Mulligan and mana-payment decisions count too (e.g. tapping the wrong land so a second spell cannot be cast that turn).

Reply with only a JSON object, no prose around it:
{{
  "summary": "two or three sentences: how the game went and the most important problem",
  "findings": [
    {{
      "decision": "d123 (the decision id where the mistake happened, or the decision right after an engine event)",
      "seat": 0,
      "cause": "one of the keys above",
      "severity": "1 minor | 2 costs material | 3 likely changed the result",
      "confidence": 0.0,
      "what": "what happened",
      "better_option": "the option index of the better play at that decision, as an integer, or null (always null for engine_bug / masking_gap)",
      "missing_play": "for masking_gap: the legal play that was not offered; else null",
      "evidence": "the log lines, probabilities, card text and rule that support the finding",
      "fix": "what in the setup to change or test (engine rule + card, mask, reward, more training on X, feature)"
    }}
  ]
}}"""


# What the policy observes (feature sets 2-3, entity trunk). Shown with
# --focus setup so the reviewer can name a missing feature, not guess one.
OBSERVES = """# WHAT THE POLICY OBSERVES (feature set 2-3, entity trunk)
The network never sees card text. It knows card names only from experience. It sees:
- Global state: step, active player, turn, each side's life, library size (bucketed), mulligans, graveyard and exile (card names with counts), mana pool, known library cards with their position, its own hand (names with counts), the opponent's hand size and any of the opponent's cards it has seen, per-side counts of permanent types (tapped and untapped), total board power per side, stack size. Per side, combat readiness: the power that could attack right now (untapped creatures that can attack), the part of it no untapped enemy creature can block (by flying/reach only), the number of untapped potential blockers, `lethal_on_board` / `evasive_lethal_on_board` (that power >= the other side's life), untapped mana sources. Deck markers.
  Note: the readiness features count only creatures that could still attack, so during the opponent's combat (attackers tapped) they no longer show the incoming damage.
- One entity per permanent and per stack item, encoded independently: name, controller, types, keywords, tapped, summoning-sick, attacking, token, `blocking` (a flag only: NOT which attacker), `attached` (a flag only: NOT to what), power / toughness / damage / counters, skip-untap, who targets it with what; stack items also have position, kind, X and what kind of object they target.
  Nothing marks an attacker as blocked or unblocked, or says how much combat damage will get through.
- Each option: the decision kind and the option's key, written with card NAMES (e.g. `declare_blocker|block|Eldrazi Spawn|Cryptic Serpent`), a pointer to the entities its label names (so two same-name objects are told apart only by their entity features), and engine previews: creatures a sweeper would kill per side, whether damage is lethal to a target, ward on a target and whether it can be paid, mana and colours left after paying a cost.
- Memory: a GRU carries state from one of the player's decisions to the next. The actions since its previous decision (its own, and the opponent's public ones, as option keys) are fed in as events (at most 256 tokens).
- Network: the entity vectors are summed into the state. In this checkpoint there is no attention between entities, so any relation between two objects has to come from features, option pointers or memory.
"""

FOCUS_SETUP = """# FOCUS
This review is for finding flaws in the SETUP before looking at play quality: engine bugs, masking gaps, observation gaps (information the policy does not get, judged against the sheet above) and architecture limits. Report misplays mainly when they point to one of these. A misplay explained by sampling or plain undertraining is only worth a finding when it is severe or recurs. For every observation_gap or architecture finding, name the exact missing feature or relation and the decision kind it affects."""


def system_prompt() -> str:
    return SYSTEM.format(causes="\n".join(f"- {k}: {v}" for k, v in CAUSES.items()))


def user_prompt(transcript: str, part: tuple[int, int] | None = None, of: int = 1, focus: str = "all") -> str:
    scope = OBSERVES + "\n" + FOCUS_SETUP + "\n\n" if focus == "setup" else ""
    if part is not None and of > 1:
        scope += f"This is the part of the game from turn {part[0]} to turn {part[1]} ({of} parts in all); review only decisions in it.\n\n"
    return scope + transcript


def parse_findings(text: str) -> dict:
    """The JSON object in a reply (tolerates code fences and surrounding text)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    out = json.loads(m.group(0))
    out.setdefault("findings", [])
    for f in out["findings"]:
        if f.get("cause") not in CAUSES:
            f["cause"] = "unclear"
        d = str(f.get("decision", ""))
        m = re.search(r"\d+", d)
        f["decision"] = int(m.group(0)) if m else None
        bo = f.get("better_option")
        f["better_option"] = int(bo) if isinstance(bo, (int, float)) or (isinstance(bo, str) and bo.strip().isdigit()) else None
    return out
