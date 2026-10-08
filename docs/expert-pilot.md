# Kalikaiz narrated-game pilot — 2026-10-08

The extraction and scenario/learning tooling is implemented. A real round was
processed, and one opening scenario with two observed action labels was
reviewed and verified in both engines. Full-round manual validation, expansion
and training are still outstanding; these counts are not a strategic dataset.

## Source and processing

[Video](https://www.youtube.com/watch?v=7rs5JfA0gnM): *DELVER NOT REQUIRED!
Pauper Mono U Terror is a Powerhouse just with Snakes and Birds*. The Jund
Wildfire round is 22:39–32:10 (571 seconds). Acquired 1080p video, English
auto-captions and metadata; linked decklist is
[7547595](https://www.mtggoldfish.com/deck/7547595). No matching native MTGO log
was available. The reviewed log crop is `(1536,120,1912,524)` for this layout;
do not assume that crop in another video.

Media and generated datasets are in `.context/expert-pilot/` locally and
`~/mtg-ml-expert/20261008-manila/pilot-v2/` on h100-private, outside Git.
The source clip's `media_offset` is 1359; all evidence uses original timestamps.
Full frame sets remain on the processing host. Some copied raw evidence
references therefore point to host paths. Review source availability before
moving these datasets to another machine.

Measured on **h100-private, NVIDIA H100 80 GB**; tools lived in isolated
Python 3.11 environments, separate from RL. Vision used vLLM 0.31.0; OCR used
PaddleOCR 3 / Paddle GPU 3.3.1. GPU UUIDs were selected from current availability.

| Stage | Actual output | Measured runtime |
|---|---|---|
| 2 FPS decoding / changed-panel OCR | 1,142 frames, 271 OCR windows | OCR loop 66.84 s on one H100; excludes OCR cold initialization and decoding |
| faster-whisper `large-v3`, word timestamps, VAD | 97 speech segments | 44.69 s on one H100; includes model loading, excludes audio decoding |
| English rolling-caption parsing | 231 segments | Not separately timed |
| Merged evidence | 390 pending log records, 328 pending narration records, 6 explicit gaps | 724 records total |

These are elapsed processing times, not measured CUDA kernel time or a
cost-per-accepted-example benchmark. Media acquisition, model downloads,
vision startup and manual review are additional. Correction minutes were not
instrumented during this first engineering pass.

## Small manually checked comparison

The same 23:04 five-card hand crop was passed through both local models with
the same structured prompt. Gold: Cryptic Serpent, Counterspell, Counterspell,
Ponder, Tolarian Terror.

| Model | GPUs | Hand result | Warm request time | Tokens (input/output) |
|---|---:|---|---:|---:|
| Qwen3-VL-8B-Instruct | 1 H100 | 4/5 card occurrences; called Ponder “Wander” | 1.81 s | 922 / 142 |
| Qwen3-VL-32B-Instruct | 2 H100s | 5/5 card occurrences, exact hand | 3.07 s | 922 / 145 |
| GPT-6.1 Sol frames | Hosted | Adapter tested; no live request, API key unavailable | Unmeasured | Unmeasured |
| Gemini 3.8 Flash video/frames | Hosted | Adapter tested; no live request, API key unavailable | Unmeasured | Unmeasured |

Full-frame attempts on both Qwen sizes produced invalid nested JSON or
invented citation intervals; the importer rejected them. A tracked full-frame
8B failure used 2,514 input / 1,427 output tokens in 11.68 s; the corresponding
32B failure used 2,514 / 1,420 in 22.72 s. Earlier development retries were not
fully instrumented. Cropping **and an explicit JSON-value example** changed
the prompt between those failures and the successful hand runs, so this is
not an isolated crop-effect experiment. Neither result estimates general
visible-state accuracy; one crop is insufficient to select a production model.

The five opening log events first visible from 22:58–23:01 were manually
annotated. Ordered exact-text scoring was **4/5 precision and recall**: OCR
joined `saidin.raken` and `skips` without a space. The event meaning survived,
but the strict text metric counted the mismatch. This tiny sample does not
establish full-round event precision/recall or the ≥95% acceptance target.

## Reviewed opening

At 22:59, saidin.raken has Cryptic Serpent, two Counterspells, Island, two
Ponders and Tolarian Terror. Both players are at 20 life with 53-card libraries
and seven-card hands; the battlefield is empty. The log confirms the fresh
game and first player's skipped draw. At 23:01 the log shows Island followed
by Ponder. All opening cards are supported by this engine.

`reviewed-opening.evidence.json` and `reviewed-opening.scenario.json` record
the source frames, review rationale and **explicit hidden fillers**. The
scenario has two action labels and stops before the engine's mana-payment
prompt. It invents no passes, mana clicks, Ponder ordering, shuffle choice or
draw. Identical Ponder copies use a normalized setup identity, without claiming
to recover MTGO's object ID. The labels mean “the player chose this”, not
“this is uniquely optimal”. This simple opening is a pipeline smoke test,
not two reviewed high-value strategic decisions.

Python and Rust produce identical player observations, legal options, action
resolution, final assertions and exported feature/history rows. Both replay
exports are viewer-compatible and hide synthetic opponent cards.
`reviewed-opening.verify.json` records the lockstep check. No recurrent history
before the opening main phase is claimed: the episode explicitly resets memory.

## Failures and remaining acceptance work

* The round's initial viewport still contains the preceding match's log.
  First-seen time cannot locate those old events in this round; they remain
  unaccepted. The merger now marks that uncertainty on initial-viewport records.
* ASR renders “Deem Inferior” as “demon fear”. A reviewed alias can aid card
  matching without rewriting the transcript or converting a prediction into fact.
* Six gap records and every log/state contradiction require manual review.
  Fixed-layout OCR can miss short dialogs and cuts; opening-state success
  does not establish that later positions are executable.
* Qwen 8B's “Wander” output passed the extraction format, illustrating why
  format validity/confidence cannot substitute for card validation and review.
  It remains pending and would fail supported-card scenario compilation.
* No game-log completeness claim, full-round accepted log, 100-decision yield,
  20-scenario yield, held-out training result or improved bot-strength claim
  is made. Three-video expansion needs reviewed complete visible states and
  whole-video holdout groups. A one-video dataset is rejected by training.

Validation of the code: focused adapter/extraction/learning and both-engine
scenario tests; the fast suite; lint; and 2,000 differential games with zero
divergences. Unit-test Bolt scenarios are explicitly synthetic fixtures and
are not included in the real pilot yield. Use [the workflow](expert-data.md)
to continue annotation and run the hosted comparison once credentials exist.
