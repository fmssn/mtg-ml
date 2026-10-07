"""LLM post-hoc review of played games, aimed at diagnosis.

A reviewer model (DeepSeek or Claude) reads a game transcript in which every
decision lists its full legal option set (the action mask: the engine only
offers options that can be completed legally) with the policy's probability
for each, plus the value estimate, the omniscient state and the engine log.
It flags apparent misplays and attributes each to a cause in the setup:
engine bug, masking gap, reward shaping, undertraining, sampling noise or an
observation gap. Claims about a better option are then checked by paired
counterfactual rollouts from the forked game (`verify`), and the reviewer's
recall is measured on games with seeded faults (`faults`).

    python -m mtg_ml.review record --agents model:runs/x/model.pt,model:runs/x/model.pt --games 5
    python -m mtg_ml.review review reviews/games/*.json --backend deepseek
    python -m mtg_ml.review verify reviews/games/<game>.json
    python -m mtg_ml.review calibrate --agents model:runs/x/model.pt,model:runs/x/model.pt

See docs/game-review.md.
"""
