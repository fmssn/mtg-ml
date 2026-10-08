# Blue benchmark specialist

`benchmark-blue@1` is a deterministic preboard Mono Blue Terror pilot for the
[benchmark plan](benchmark-plan.md), using the
[benchmark foundation](benchmark-foundation.md). It supports the frozen 60-card
Blue list and both physical seats. Development results compare this pilot with
the unchanged legacy Blue pilot against fixed legacy Jund and Blue opponents.
They do not certify the full benchmark release or general Pauper proficiency.

## Registration and information

```python
from mtg_ml.benchmark import AgentRegistry, ScriptedAdapter
from mtg_ml.benchmark.blue import BOT_ID, register_blue

registry = AgentRegistry()
register_blue(registry, source_revision="<full committed source revision>")
agent = ScriptedAdapter(registry.create(BOT_ID), seat=0)
agent.reset(own_deck)
```

The specialist consumes detached `PlayerView` and `LegalAction` data. Its
information contract is `own-list-hidden-opponent-v1`; registration records rule
revision 1 and the canonical digest of the immutable `PARAMETERS` map. Its
`observe` hook validates event perspective without building inferred private
history. Current engine knowledge supplies every decision, including reset-mode
positions. Own-list map insertion order does not affect decisions.

The adapter supplies structured stack IDs, relative controllers, targets, cast
methods, X, public card/source references and ward's affected spell and numeric
amount. Stack-only card rules are visible. It does not forward raw trigger data,
engine objects, callbacks, opponent private menus or unknown library order.
Strategy never parses prompts or display labels. Scores tie at the lowest current
index. Unsupported decision kinds, action verbs and casting modes raise errors.

## Rules and parameters

The rules cover Island, Delver, Terror, Serpent, all four cantrips, Lorien,
Counterspell, Force Spike, Deem Inferior, Sleep and Plunder, including flashback
and escape. Forced discard, sacrifice, ordering, X, trigger ordering and dungeon
choices have explicit generic handlers.

- Keep two to four Islands, or one Island supported by at least two cheap
  cantrips; keep at four cards to bound mulligans. Bottom redundant mana and
  expensive threats using current hand and board needs. Seek four total lands.
- Prefer cheap pressure, cast cantrips to find action and fuel discounts, cycle
  Lorien when mana is missing and use spare mana at the opponent's end step.
  Preserve Counterspell's actual `{U}{U}` when protecting pressure against a
  nonempty hand; rebuild pressure rather than holding mana on an empty board.
- Put the least useful Brainstorm cards back, with the better of those cards
  ending on top. Order Ponder by current value and shuffle unsuitable cards,
  including landless selections while short of mana. Do not self-mill known
  cards valued at least 3.5, or draw beyond the remaining library. Thought Scour
  can target the opponent when self-mill would cause decking.
- Counterspell's minimum threat value is 2.5; Force Spike's is 1.5. Choose actual
  stack IDs, account for higher counters and cumulative ward payments, and
  prefer a live cheaper Spike. Shaman plus Toxin is a wipe threat to ground
  creatures, including damage already on the stack; it does not threaten a board
  consisting only of flyers.
- Use bounce and tap for lethal, survival or productive pressure. Include ward
  in the mana budget, avoid refreshing an existing tap duration, and retain
  discount fuel when escape is merely a tempo play. Lethal overrides that fuel
  preference. Recover useful own threats after Deem rather than burying them.
- Combine attack pressure and remaining defense. Public blocking assignment
  computes guaranteed player damage, including cumulative trample prevention.
Damage splits and sequential allocation share the victim-value and lethal
  objective; the sequential suffix optimization costs O(blockers × power).

Combat remains a heuristic pilot: it does not simulate future game branches or
unknown combat tricks. Blocking optimization for these decks is exact for at most twelve
attackers with at most one trampler. Wider boards use a conservative prevention
bound without dropping entities or legal actions. Attack and block trade scores
also use public power, toughness, damage, evasion, deathtouch, lifelink and
trample. The source revision and parameter digest freeze these judgments.

## Reviewed development fixtures

The cases in `tests/test_benchmark_blue.py` are implementation-reviewed synthetic
positions with explicit rationale and complete-line assertions. They are not
human demonstrations or independently certified final puzzles. Stacks and
floating mana are constructed through legal actions. The allocation fixture's
non-Blue Hydra deliberately checks the protocol's generic combat handler.

| Area | Favorable and unfavorable cases |
|---|---|
| Opening and mana | Supported and unsupported one-land keeps; excess lands; colored reservation versus deployment; floating payment before another source; Lorien cycling and drawing |
| Library choices | Distinct Brainstorm put-backs, witnessed second-card-on-top order and subsequent milling; Ponder ordering and shuffling; Delver reveal; known-top preservation; safe versus fatal self-mill |
| Stack interaction | Harmless artifacts; payable and unpayable ward/Spike; combined taxes; duplicate-name counter wars; Shaman–Toxin ground wipe versus unaffected flying Delver |
| Resources and tempo | Discounted deployment; escape preserving fuel versus taking lethal; Plunder normal and flashback outcomes; own Deem placement; tap duration and evasive lethal |
| Combat and protocol | Combined lethal through a blocker; defensive blocking; eleven-blocker allocation; cumulative trample prevention and a second block that prevents lethal without killing the attacker; hidden resampling, metadata/label independence, reset isolation, map ordering, stable ties and explicit unsupported behavior |

Both-engine fixture checks include target selection, every offered payment and
choice, and the resulting public state. The development comparison tests reject
missing, duplicate or mismatched rows, technical errors and invalid outcomes.
Opposite per-cell effects test shared bootstrap indices rather than independent
resampling. Complete games compare every specialist view, legal action and
selected index between Python and native against both opponents in both seats.
The multi-game parity and mode checks are marked slow; a small fast fixture also
checks deterministic mode and actor-seed independence.

## Reproduce development validation

Use this checkout's environment and native extension:

```bash
make native
.venv/bin/python -m pytest -q tests/test_benchmark_blue.py tests/test_benchmark_blue_comparison.py
make test-fast
make lint
git diff --check

.venv/bin/python tools/benchmark_blue.py --phase verify --engine native \
  --workers 4 --out .context/blue-parity
.venv/bin/python tools/benchmark_blue.py --phase smoke --engine native \
  --workers 4 --out .context/blue-smoke
.venv/bin/python tools/benchmark_blue.py --phase compare --engine native \
  --workers 4 --out .context/blue-comparison
```

Outputs must be new directories and source must be committed. `freeze.json`
records source files, parameters, native build, card specification, exact deck
maps and legacy sources pinned at `66345da4c5bff4a1a31ab0047ff8b1abf3869c49`.
The Blue and Jund deck hashes match PR62. `planned.json` fixes all row identities.
Atomic batch files retain outcomes, decision counts, timing and failure details;
errors get reproducible replay files. A changed source invalidates completion.

The smoke panel has 40 blocks per cell, 320 candidate games. The paired pilot has
400 blocks per cell, 3,200 games per arm and 6,400 games total. Both use the
`benchmark-v1/dev` stream, four seat/start slots, 100 turns, a 10,000-decision cap,
and disabled single-choice, mana and pass shortcuts. Scripted play is deterministic,
so the campaign records greedy mode; sampled-mode equivalence is tested separately.

`report.json` uses `BlueSpecialistDevelopment` version 1, separate from the
foundation's checkpoint-only `BenchmarkResult`. The candidate uses fair inputs;
legacy opponents and baseline are diagnostic adapters. This mixed development
panel does not receive a fair full-suite aggregate.

Each block averages its four outcomes (win 1, draw ½, loss 0). The comparison
joins identical planned rows and bootstraps 10,000 whole blocks, sharing sampled
indices across cells. Report each cell and the equal-weight mean with 95%
percentile intervals. Stronger play requires a mean lower bound above zero and
no cell with an interval upper bound below −0.03. A single block has no interval.

## Validation results

Measured on 2026-10-08 on Apple M3, macOS 26.6.2, Python 3.11.11, with native
execution and four workers. The final implementation freeze is
`90dacdbdc02982efe615180f683188fe6340a5f9`; subsequent changes add validation
fixtures and this report. Legacy sources remain frozen at
`66345da4c5bff4a1a31ab0047ff8b1abf3869c49`. The parameter digest is
`f80cb31cad22feaa42a5d1488356151fa4ce4f36f9b02a27922212a156b4c68c`.
Exact deck maps, native/card/source hashes, planned/completed identity hashes,
raw-file references and timing summaries are in
[the compact result summary](benchmark-blue-results.json).

The 320-game legality panel and 6,400-game comparison completed with zero
technical errors, missing rows or duplicate identities. No game ended in a draw.
The eight-game matched subset was identical across Python/native and one/four
workers. Separate complete-game tests compared every specialist view, structured
legal action and selected index across engines; sampled/greedy choices matched.

| Opponent | Candidate score | Legacy Blue score | Paired difference | 95% interval |
|---|---:|---:|---:|---:|
| Legacy Blue | 72.13% | 50.00% | +22.13 pp | +19.88 to +24.31 pp |
| Legacy Jund | 80.00% | 68.44% | +11.56 pp | +8.94 to +14.25 pp |
| Equal-weight mean | 76.06% | 59.22% | **+16.84 pp** | **+15.06 to +18.63 pp** |

Each cell has 400 blocks and 1,600 games per arm. An independent strict rejoin
and 10,000-replicate shared-index bootstrap reproduced the report exactly. The
development strength gate passed: the mean's lower bound exceeds zero and both
cell intervals are positive. This supports stronger play against these fixed
legacy opponents; it does not certify final puzzles or the release benchmark.

The smoke took 56.07 seconds and the comparison 1,049.56 seconds. Candidate
adapter decision latency during comparison was 0.422 ms median and 1.652 ms at
the 95th percentile. These are local measurements, including view translation;
the project test suite ran concurrently.

- Focused specialist/comparison validation: **114 passed, 1 expected skip**,
  including slow complete-game checks.
- Full `make test-fast` on the final implementation: **1,590 passed, 16 skipped,
  21 slow tests deselected**. The additional crowded-allocation and complete
  view/choice fixtures were also checked in the focused suite above.
- Workspace native extension rebuilt; `make lint` and `git diff --check` passed.
  No engine rules, existing golden digests, legacy pilots or evaluation behavior
  changed. CI runs both 300-game differential-fuzz panels as well.

Final raw artifacts are `.context/blue-acceptance-{parity,smoke,comparison}/`;
logs are `.context/blue-acceptance-{focused,fast}.log`. Earlier campaigns,
including the completed `b2eeffb` panel, are superseded and excluded from these
statistics. The final correction covers an additional trample block needed for
survival; its regression demonstrably fails on the previous rule. All retained
results use the corrected source. PR65 remains draft until the user confirms
readiness. Shared view and test-registration reconciliation with Jund belongs
to the foundation integration owner.
