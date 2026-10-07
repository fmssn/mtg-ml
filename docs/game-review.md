# LLM game review

`mtg_ml.review` has an LLM read played games and flag misplays. Its main job is to trace each misplay to a flaw in the setup: an engine bug, a masking gap, reward shaping, undertraining, sampling noise or an observation gap. It never trains anything. Its output is a list of findings to check and fix.

## Why the transcript holds what it does

| in the transcript | lets the reviewer tell apart |
|---|---|
| every option the engine offered, at every decision with a choice | "the better play was not offered" (masking gap, or an engine legality bug) vs "offered but not taken". The engine only offers options that can be completed legally, so this list is the action mask. |
| the policy's probability for each option | sampling noise (a low-probability option was sampled) vs a policy that is confident in the wrong play (undertraining or shaping) |
| the value estimate per decision | whether the policy "knew" a choice was bad (value drops right after it) |
| card text as the engine implements it, and the log | engine bugs: the log and state checked against the text and the rules |
| the checkpoint's training state and the training reward | undertraining vs reward shaping (life-difference shaping while it was annealed) |
| the omniscient state, with each seat's hidden information marked | judging a decision only on what that player could know |

Forced decisions (a single option) appear only as log lines. A state line is printed only when it changes. A 15-20 turn game comes to about 25-45k tokens.

## Pipeline

```bash
M=model:runs/<run>/model.pt
python -m mtg_ml.review record --agents $M,$M --games 20 --engine native   # reviews/games/*.json
python -m mtg_ml.review review reviews/games/*.json --backend deepseek      # *.findings-deepseek.json
python -m mtg_ml.review verify reviews/games/*.json --backend deepseek      # rollouts for "B was better" claims
python -m mtg_ml.review report reviews/games/*.json --backend deepseek > reviews/report.md
```

- **Backends.** `deepseek` (`DEEPSEEK_API_KEY`; `--model deepseek-reasoner` for the thinking model) is the cheap first pass. `claude` (the `anthropic` package and its usual credentials; default `claude-opus-5-5`, `--model claude-sonnet-5-5` is cheaper) is for escalation: rerun it on games where the first pass found engine bugs, masking gaps or severity-3 findings. On a safety decline the request falls back server-side (`fallbacks: "default"`). `prompt` writes `<game>.prompt.md` and makes no call: any agent can answer it, for example a Claude Code subagent. Save the reply as `<game>.reply.txt` and read it back in with `--backend file`.
- **Long games** are split by turns to fit the backend's context (`MAX_CHARS`). The system prompt is the same for every game, so it is served from the prompt cache.
- **Verify.** The game is rebuilt from its recorded constructor arguments and replayed to the decision. The recorded agents replay the prefix too, so GRU memory matches the real game. For each of `n` determinizations (everything the deciding seat could not see is re-dealt), both the chosen option and the claimed better one are played out with the same seeds. The verdict uses the mean paired difference in the seat's result and its standard error: `confirmed` (> 2 se and >= 0.05), `refuted`, or `inconclusive`. Rollouts use the same policy, so "better" means better for this policy's own continuation. On the Python engine, 16 playouts take about 2 s. Engine-bug and masking-gap findings get no rollout; they need a rules test (`docs/adding-cards.md`).

## Calibration: seeded faults

The reviewer can't be trusted until its recall on faults you planted is known:

```bash
python -m mtg_ml.review calibrate --agents $M,$M --games 10 --backend deepseek
python -m mtg_ml.review report reviews/games/*-faults.json --backend deepseek --calibration
```

The faults are defined in `mtg_ml/review/faults.py`, one per cause the reviewer must recognise:

| fault | simulates | cause expected |
|---|---|---|
| `mask:<seat>:<text>` | options whose label contains `<text>` are never offered to that seat (default `mask:0:blocks`: Jund never gets to block) | masking_gap |
| `blunder:<seat>:<n>` | up to `n` confident decisions replaced by the policy's least likely option | sampling_noise |
| `life:<seat>:<turn>:<delta>` | the recorded life total drifts by `delta` from that turn on, with no event | engine_bug |
| `power:<seat>:<turn>:<delta>` | one creature's recorded power drifts | engine_bug |

A fault counts as caught when a finding with the right cause cites a decision within 3 of one where the fault acted. An engine-state fault also counts at any later decision, because the fault persists. A mask fault also counts when the finding names the hidden text. Findings that match no fault are counted as well, as a noise measure. Option faults run on the Python engine only, and their games cannot be verified, because the recorded choices index the filtered option lists.

Some blunders are harmless: the least likely option can be as good as the chosen one. Read missed blunders before you blame the reviewer.

## Results

See the PR that added this module for the first calibration run and the findings on `v3-entity-500k` games.
