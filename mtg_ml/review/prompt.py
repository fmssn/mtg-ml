"""The reviewer's instructions, the cause taxonomy and the findings format."""

from __future__ import annotations

import json
import re

from ..encode import check_features

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


# Additions are cumulative. Keep this reference aligned with docs/features.md;
# select the version from the recorded checkpoint, never the current default.
FEATURE_NOTES = {
    1: """Global state includes step, active player, turn, life, library/hand/zone sizes, mulligans, own hand names, known opponent cards, graveyard/exile names, floating mana, board type counts and power, stack size and known library cards. Hidden opponent cards and library order are not inputs. Permanent entities have name, controller, types, keywords, tapped/sick/attacking/token/blocking/attached flags, P/T, damage and counters; stack entities identify the item. Options carry decision kind and structured action-key tokens, with entity pointers for objects they refer to. The network does not read raw oracle text.""",
    2: """Adds readiness/evasive-power, potential blockers, lethal-on-board and untapped-mana state features; known library positions; entity skip-untap, stack targets, targeting relations and X. Static option previews describe sweeper kills, target ward/payability and lethal damage, mana and colours left after a cost. Adds the self deck marker. Ready power is not itself an incoming-combat-damage measurement.""",
    3: """Adds the opponent deck marker (`opp:deck:`), allowing deck-conditioned matchup mixes.""",
    4: """Adds combat relations: attacker slots, blocked/unblocked flags, blocker counts/power/lethality, and which attacker a blocker blocks (slot/name/characteristics). Global inputs include attacking and unblocked power, incoming lethal and life after unblocked damage. Block previews describe attacker/blocker death and lethal damage still unblocked. X has numeric/max/entering-power previews and an option-to-spell pointer. Mana colour supply/hand needs and colour previews cover land plays, basic searches and payments.""",
    5: """Adds card-shape tokens derived from the structured card specification: costs, ability/trigger/effect ops and target kinds. These describe what cards do without giving the model raw card text. The decider's own hand cards are entities, and cast/play-land/plot options point at them; stack items include their resolving ops.""",
    6: """Adds simulated option previews (`pv:sim:*`) from applying an option to an exact game copy, plus assume-opponent-passes previews (`pv:simp:*`). Observable deltas include life, creatures/permanents gained/lost, zones, mana/colours and lethal flags. Simulations stop at hidden information, relevant decisions, game end or a bounded horizon and can be skipped; they are not unrestricted search or knowledge of hidden draws. Existing static previews remain.""",
    7: """Hidden-list information contract: no archetype labels (`self:deck:*`, `opp:deck:*`) and no absolute seat token. The viewer's own registered main, registered sideboard and current-main counts are inputs (`self:list:*`, every copy, hashed). Entity lists are no longer capped: every battlefield object, stack item and own-hand card keeps its action pointer. Every candidate option receives both simulated preview streams, however many candidates there are.""",
    8: """Set 7's hashed features unchanged, plus exact witnessed opponent-card evidence for the belief head: what this player actually saw of the opponent's cards in the current game and in each finished game of the best-of-three. The evidence feeds the belief head, not the hashed inputs; hidden cards and library order are still not observed.""",
}


def observation_sheet(config: dict | None = None) -> str:
    """Describe an actual policy, or give a versioned reference for legacy files."""
    if config is None:
        lines = [
            "# WHAT THE POLICY OBSERVES (configuration unknown)",
            "This replay has no structured policy configuration. The following is a cumulative versioned reference, not a claim that this checkpoint uses the latest features. Resolve its features, trunk, memory and entity_attn from the original checkpoint before confirming an observation or architecture gap.",
        ]
        versions = FEATURE_NOTES
    else:
        features = check_features(config.get("features", 1))
        trunk = config.get("trunk", "mlp")
        memory = config.get("memory", "gru")
        attention = config.get("entity_attn", 0)
        lines = [f"# WHAT THE POLICY OBSERVES (feature set {features})",
                 f"Recorded architecture: trunk={trunk}, memory={memory}, entity_attn={attention}."]
        versions = range(1, features + 1)
        if trunk == "entity":
            lines.append(f"Entity vectors use {attention} self-attention layers before being summed into the state; option pointers use the corresponding entity vectors.")
        else:
            lines.append("This is not the entity trunk. Check mtg_ml/rl/model.py for how this trunk consumes state/entity tokens and option pointers; do not assume entity self-attention.")
        if memory == "gru":
            lines.append("A GRU carries recurrent state between the player's decisions; event tokens describe own and public opponent actions since the previous decision (at most 256 tokens).")
        else:
            lines.append("There is no recurrent memory between decisions; current event tokens still describe own and public opponent actions.")
    lines += [f"- Feature set {v}: {FEATURE_NOTES[v]}" for v in versions]
    capped = "Entity lists are capped at 64 (feature sets below 7). " if config is None or features < 7 else ""
    lines.append("Numbers use hashed/bucketed/thermometer representations, not raw arithmetic. " + capped + "Feature presence does not prove the policy learned to use it. For a claimed gap, inspect the checkpoint-version inputs at the cited decision against docs/features.md, mtg_ml/encode.py and mtg_ml/rl/features.py.")
    return "\n".join(lines) + "\n"


# Retain a general reference for callers that imported the old constant.
OBSERVES = observation_sheet()

FOCUS_SETUP = """# FOCUS
This review is for finding flaws in the SETUP before looking at play quality: engine bugs, masking gaps, observation gaps (information the policy does not get, judged against the sheet above) and architecture limits. Report misplays mainly when they point to one of these. A misplay explained by sampling or plain undertraining is only worth a finding when it is severe or recurs. For every observation_gap or architecture finding, name the exact missing feature or relation and the decision kind it affects."""


def system_prompt() -> str:
    return SYSTEM.format(causes="\n".join(f"- {k}: {v}" for k, v in CAUSES.items()))


def user_prompt(transcript: str, part: tuple[int, int] | None = None, of: int = 1, focus: str = "all", policy_configs: list[dict | None] | None = None) -> str:
    scope = ""
    if focus == "setup":
        sheets = [f"# POLICY P{i}\n{observation_sheet(c)}" for i, c in enumerate(policy_configs or ()) if c is not None]
        scope = "\n".join(sheets or [OBSERVES]) + "\n" + FOCUS_SETUP + "\n\n"
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
