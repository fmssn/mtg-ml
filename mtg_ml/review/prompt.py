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


def system_prompt() -> str:
    return SYSTEM.format(causes="\n".join(f"- {k}: {v}" for k, v in CAUSES.items()))


def user_prompt(transcript: str, part: tuple[int, int] | None = None, of: int = 1) -> str:
    scope = ""
    if part is not None and of > 1:
        scope = f"This is the part of the game from turn {part[0]} to turn {part[1]} ({of} parts in all); review only decisions in it.\n\n"
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
