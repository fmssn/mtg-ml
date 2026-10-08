# Narrated MTGO evidence, scenarios and demonstrations

`python -m mtg_ml.expert` implements the extraction → review → legal scenario →
expert-feature pipeline. Media dependencies are optional and isolated from RL.
See [the measured pilot](expert-pilot.md) for actual yield and remaining work.

## Files and knowledge boundaries

| Artifact | Contract |
|---|---|
| `EvidenceGame`, version 1 | Sources, original-time references, partial facts, narration, gaps and independent review decisions. No engine state is implied. |
| `ExecutableScenario`, version 1 | Accepted initial state, field-by-field evidence or named synthetic completions, legal setup actions, stable action selectors, assertions and observed demonstrations. |
| `ExpertEpisode`, version 1 | Existing policy features and legal-option rows, evidence IDs, video group and an explicit memory reset. No narration is a policy input. |
| Replay, format 1 | Existing viewer-compatible **reconstruction** with assumptions. It cannot restore arbitrary engine state. |

Each record has `id`, `kind`, `value`, `visibility`, `available_at`, `refs` and
`review`. A reference has `source_id`, original `start`/`end` seconds, source
`kind`, `locator` and optionally `viewer`. Accepted reviews need reviewer and
rationale. Extraction and model interpretation always produce pending records.
JSON writes are atomic; reviewed extraction files are never overwritten by a rerun.

State facts use `value: {"path": "players.0.hand", "value": ["Island", "Ponder"]}`.
Visibility is public, self, opponent_private, belief, hindsight or unknown.
Commentary separately classifies observed actions, proposed lines, rejected
alternatives, opponent predictions and retrospective corrections. Card/time
alignment supplies ranked candidates for review, never automatic approvals.
Use an aliases JSON map, for example `{"demon fear": "Deem Inferior"}`; the raw
transcript is preserved. Captions and ASR remain separate sources of evidence.

Compilation rejects future state references, private opponent facts, beliefs,
hindsight, pending facts, unsupported cards, contradictory field bindings and
ambiguous action selectors. Every bound field must equal the reviewed fact's
value. Public state and the player's own hand cannot be synthetic in a
demonstration. Hidden fillers must preserve observed hand/library counts.
Later action references can label a decision; they cannot supply its inputs.

## Acquire and extract

Keep videos, decoded frames, datasets, API responses and checkpoints outside Git
(for local work, `.context/expert-pilot/` is already ignored). Use authorized
recordings or the authorized video source. An original recording or matching
MTGO log is preferable when available.

```bash
make setup
.venv/bin/python -m pip install -e '.[expert-media]'
# ffmpeg must be on PATH
.venv/bin/python -m mtg_ml.expert acquire \
  'https://www.youtube.com/watch?v=7rs5JfA0gnM' --out .context/expert-pilot/source
.venv/bin/python -m mtg_ml.expert extract \
  --source .context/expert-pilot/source/source.json \
  --start 1359 --end 1930 --fps 2 \
  --captions .context/expert-pilot/source/video.en.vtt \
  --out .context/expert-pilot/extracted
```

Acquisition preserves full-resolution video, English captions, description,
chapters and decklist links. A manually acquired clip can use `media_offset`
in its source JSON; extracted references still use original video seconds.
Choose reviewed game intervals from chapters; deck tech, sideboarding and
edited gaps require their own segments. First-seen OCR time is **not** the
actual event time. In particular, the initial viewport can contain a previous
game. Reject or separate such entries during review.

Add `--ocr --ocr-device gpu:0` in an OCR environment, and
`--asr --asr-device cuda --asr-model large-v3` in an ASR environment. The crop
argument is `--crop x,y,width,height`. Without it, detection searches log-text
clusters and the light panel surface; uncertain layouts fail with a request
for a reviewed crop. A fixed crop remains valid only within that layout.
Changed panels are OCRed; overlapping scrolling sentences are merged by
occurrence, preserving repeated identical actions. Missing overlap and
occlusion become gap records. `merge_log` also accepts explicit cut markers
from reviewed frame windows. Wrapped timestamped sentences are joined.

`frames.json`, raw `ocr-windows.json` and word-timestamped ASR support auditing.
The decoder caches frames only for identical media size/mtime, interval,
offset and FPS. Failed OCR/ASR can reuse that cache; changed inputs or a finished
evidence file require a fresh output directory. Gaps, cuts, partial lines,
missed short dialogs and layout changes still need review; 2 FPS is a pilot
setting, not a completeness guarantee.

For local MTGO logs:

```bash
.venv/bin/python -m mtg_ml.expert import-log evidence.json game.txt \
  --source-id 7rs5JfA0gnM --start 1359 --end 1930 --out with-log.json
# For Match_GameLog_*.dat, additionally pass --parser /path/to/file_parser
# from https://github.com/BigPeet/mtgo_utils
```

Binary logs are never decoded as arbitrary text. Imported log lines have
ordered but unaligned timing; narrow their references before using them to
attest a particular position. Matching native logs were unavailable for this pilot.

## Local and hosted interpretation

Install PaddleOCR 3 and faster-whisper in a separate processing environment.
The CPU OCR extra installs `paddlepaddle`; on H100, install the vendor CUDA
wheel `paddlepaddle-gpu` instead, plus `paddleocr>=3,<4` and `setuptools>=70`.
Do not install conflicting CPU and GPU Paddle runtimes. Use a third environment
for vLLM/Qwen; never upgrade the RL environment's Torch to serve vision models.
Choose currently free GPUs by UUID and set `OMP_NUM_THREADS=8`.

```bash
# On the processing host, in its isolated vision environment:
CUDA_VISIBLE_DEVICES=<free-GPU-UUID> OMP_NUM_THREADS=8 \
  vllm serve Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port 18108 \
  --max-model-len 16384 --gpu-memory-utilization 0.75 --max-num-seqs 1
# For 32B use two free GPU UUIDs and --tensor-parallel-size 2.
ssh -N -L 18108:127.0.0.1:18108 h100-private
.venv/bin/python -m mtg_ml.expert interpret evidence.json --frames frames.json \
  --backend local --endpoint http://127.0.0.1:18108/v1 \
  --start 1383 --end 1386 --viewer 0 --out vision-evidence.json
```

For OpenAI use `--backend openai --model gpt-6.1-sol` and `OPENAI_API_KEY`.
For Gemini use `--backend gemini --model gemini-3.8-flash` and `GEMINI_API_KEY`
(or `GOOGLE_API_KEY`). Adding `--video <YouTube-URL>` uses Gemini's short video
window with original offsets and 2 FPS; otherwise all backends see the selected
frames and timestamped commentary. `--source-id` is required for multi-source
evidence. Credentials are read from environment variables and never printed.
Local endpoints must be loopback; use a tunnel for remote GPUs.

All providers use the same structured record contract. Frame citations must
equal their supplied timestamp; transcript citations must lie within the
supplied interval. Invalid locators, truncated output and invalid nested JSON
are rejected. A failed CLI attempt writes `<output-stem>.attempt.json` with
runtime, model, observed usage and failure reason. Successful usage is recorded
in evidence metrics. Monetary cost must be computed using the account's
actual pricing; tokens, GPU wall time, cold-start time and correction minutes
must be compared separately. Smaller crops materially helped the pilot.

Primary references: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL),
[OpenAI images](https://developers.openai.com/api/docs/guides/images-vision),
[Gemini video](https://ai.google.dev/gemini-api/docs/video-understanding).

## Review and compile

Open `apps/expert-review/index.html` locally. Load evidence, inspect the cited
recording/frame, edit the value/visibility/timing/classification, and save a
reviewed file or patches. Source references are displayed; the page does not
load remote media, send data to a server or execute engine code. More involved
reference corrections or new state/action records can be made in the JSON.

```bash
.venv/bin/python -m mtg_ml.expert review evidence.json review-patch.json --out reviewed.json
.venv/bin/python -m mtg_ml.expert template reviewed.json --source-id 7rs5JfA0gnM \
  --id opening --time 1379 --viewer 0 --out opening.scenario.json
```

The template leaves unavailable fields null. Fill each unknown explicitly or
keep the position as evidence. Version 1 accounts for active player, phase,
turn, lands played, spells cast, life, cards drawn, every zone and visible zone
counts. Cards can have stable IDs and battlefield tapped/sick/counter flags.
The compiler uses `setup`, `add_card`, `start_step` and legal `setup_actions`.
Damage, attachments, mana pools, known-library markers, combat and arbitrary
stack/coroutine snapshots cannot be inserted through the initial format;
construct supported positions through legal actions. Never claim this is a
general MTGO state-restoration format.
Demonstration initial libraries/opponent hands must be empty or explicit
hidden completions. Observed knowledge of those zones must be recreated by
legal setup actions; binding a visible opponent hand would otherwise silently
hide it from the policy. A partially known hidden zone can remain evidence
until that reconstruction is supported and reviewed.

Selectors have `player`, decision `kind`, exact option `key` or `label`, and
optional `objects` containing stable initial card IDs. Keys alone cannot
identify two copies; ambiguity is an error. The engine's current option index
is resolved at compilation, never stored as a durable action identity. For
example: `{"player":0,"kind":"priority","key":["play_land","Island"]}`.
For demonstrations, accepted log/action facts must contain the same reviewed
`selector`. Setup actions construct the position; they are not expert labels.

Use `preferred` for reviewed acceptable strategic actions, `continuation`
plus `expected_after` for rule/outcome regressions, and `demonstration` entries
`{selector, refs, label}` for observed choices. These are distinct claims.
Expected views/outcomes are path/value maps over `observe`; assertions and
selectors reject contradictions instead of falling back to another action.

```bash
.venv/bin/python -m mtg_ml.expert verify opening.scenario.json --evidence reviewed.json --out verified.json
.venv/bin/python -m mtg_ml.expert compile opening.scenario.json --evidence reviewed.json --out view.json
.venv/bin/python -m mtg_ml.expert regression opening.scenario.json --evidence reviewed.json --out result.json
.venv/bin/python -m mtg_ml.expert replay opening.scenario.json --evidence reviewed.json --out replay.json
.venv/bin/python -m mtg_ml.expert export opening.scenario.json --evidence reviewed.json --features 6 --out episode.json
```

`verify` requires the native build and compares observations and legal actions
at every declared continuation step, then checks expected outcomes. Replay
exports hide synthetic opponent hands and private opponent decision echoes.
Our continuation seed reproduces **our completion**, never MTGO's shuffle.

## Learning and evaluation

Version 1 treats scenario starts as new episodes with reset memory. Every
subsequent decision of the reviewed player participates in recurrent history,
including unlabeled decisions. Do not insert guessed passes or mana clicks
to bridge missing history. An unknown own library cannot cross a draw/new turn
or supply a private library-choice label; start a separately reviewed scenario.

```bash
.venv/bin/python -m mtg_ml.expert score parent.pt episodes/*.json --out before.json
.venv/bin/python -m mtg_ml.expert train parent.pt episodes/*.json \
  --out .context/expert-training/policy.pt --epochs 5 --lr 1e-5 \
  --heldout-fraction 0.2 --device cpu --seed 0
```

Training uses the checkpoint's architecture/features, legal-action supervised
loss and a separate optimizer. It does not inject demonstrations into PPO,
overwrite the parent, or carry stale PPO optimizer moments. Episode feature
versions must match the checkpoint. Video groups are hashed into train/test;
all match/scenario variants share their video's group. Training refuses a
dataset without both training and held-out groups. The saved report includes
before/after held-out NLL and choice accuracy, group membership and flags.
Shared policy/value representations change, so value calibration needs evaluation.

Before a real training run, follow [the experiment ledger](experiments/README.md)
with archived parent, deployed code ref and exact flags. Compare unchanged and
supervised copies on held-out tactical objectives, sampled/greedy benchmark
and L1 ladder before claiming improvement. No real training run was launched
for the single-video pilot.

`metrics --gold ordered-manual-event-texts.json [--start N --end N]` measures
occurrence-sensitive ordered event precision/recall. It accepts correction
minutes and measured API cost. Gold must cover the same first-seen window;
format validity and model confidence are not accuracy. For visible-state
benchmarks, manually compare zone/card multisets, counts and exact fields on
the same frames across providers. Track raw versus corrected results and
accepted examples per video hour. Target ≥95% event precision/recall on full
manual annotation before accepting extraction at scale. Expansion targets
remain three videos, 100 reviewed strategic decisions and 20 runnable scenarios.
