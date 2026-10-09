# Jund benchmark specialist, revision 1

`benchmark-jund@1` is an opt-in preboard specialist for the registered
`jund_wildfire` list. It implements the Jund follow-up in
[the benchmark plan](benchmark-plan.md), on the interfaces from PR64 and the
corrected combat engine inherited from PR63. `make_bot`, training opponents and
legacy strategy code are unchanged. This is a development pilot, not the frozen
benchmark release or its reserved puzzle corpus.

## Registration and information

```python
from mtg_ml.benchmark import AgentRegistry, ScriptedAdapter
from mtg_ml.benchmark.bots.jund import register_jund

registry = AgentRegistry()
register_jund(registry, tested_source_revision)
agent = ScriptedAdapter(registry.create("benchmark-jund@1"), seat=0)
agent.reset(own_registered_main)
```

The caller supplies the tested source commit. Registry metadata records deck ID,
rule revision 1, the immutable parameter digest and
`own-list-hidden-opponent-v1`. The strategy receives detached immutable views,
semantic action keys and object references. It never reads engine objects,
labels, prompts, rules text, the opponent's registration or unrevealed hand.
Publicly revealed cards and available public mana can influence choices.

`reset` replaces the own registration. Every decision constructs its analysis
from the current view; there is no action history or cached position to invent
for a mid-line start. The shared cache only parses immutable mana-cost strings.
Own registration is supplied by the adapter on every decision. Stack targets,
payments, live/last-known sources, selected public trigger fields and blocked
attacker IDs are adapter data. Arbitrary engine trigger payloads are excluded.
Ward, tapped entry and alternate costs are explicit card rules. These adapter
additions use existing engine properties; this PR changes no engine rules.

## Ordered rules

Main priorities are available lethal, avoiding immediate defeat, useful tactical
exchanges, development/card advantage and passing. Numerical priority bands,
card valuations, sacrifice values and combat weights live in the immutable
`bots/parameters.py` set and are included in the registration digest. Applicable
legacy card values were copied; the specialist does not inherit legacy strategy.

Opening hands account for colored sources, Landscape access and early spells.
Bottoming evaluates the retained hand, protecting unique colors and sufficient
lands. Land sequencing considers immediate untapped mana, indestructible
Wildfire targets and the affinity discount of even a tapped artifact land.
Basic search uses own registration minus known zones and never assumes an
opponent list. An exhausted own basic supply suppresses Wildfire ramp.

Payments assess remaining colored feasibility and a concrete useful Cast Down
reserve. Sacrifice feasibility removes a self-sacrificing mana source from the
payment supply before treating it as fodder. Munitions partitions available
Spawns between payment and fodder, allowing two Spawns to fund one shot without
counting either token twice. Lethal calculations can spend valuable creatures
and artifact lands; nonlethal exchanges retain the cheap-fodder limit. Wellspring draws, optional Spellbomb
payment, Lembas recycling and Infiltrator/Chrysalis growth affect fodder values.
Removal checks the target and ward before casting, and ignores a target already
covered by pending Cast Down. Munitions distinguishes lethal face damage from a
useful creature kill. Known, affordable Counterspell modestly favors cheaper
bait; unknown cards are never treated as counters.

Shaman–Toxin evaluates friendly collateral, flying exclusions, lifegain and
pending damage. Toxin must resolve before deathtouch is used. Multiple necessary
activations are placed before Shaman dies, with live/LKI keywords taken from the
public stack view. Every response causes a fresh calculation. Other handlers
cover Hydra normal/bestow and X, Familiar, Chrysalis/Spawn, Infiltrator, draw
outlets, graveyard denial and artifact activations.

Combat uses static simultaneous-damage arithmetic, including evasion,
deathtouch, trample, indestructible, both players' lifelink and vanished blockers.
It compares grouped attacks, individual additions and all-in-minus-one groups;
blocking considers single blocks and inexpensive lethal gangs. It conservatively
estimates crack-back from the visible opposing army. This is a bounded heuristic,
not exhaustive optimal combat or an engine rollout. Every complete damage option
is scored; PR63 sequential allocation uses suffix dynamic programming in O(nP),
with lethal ranked above material. Spawn sacrifice evaluates growth and actual
blocks after declaration rather than inventing a fresh blocking assignment.
Priority in combat-damage/end-combat steps cannot cause another damage step;
Spawn growth is not treated as immediate combat lethal after damage resolves.

## Decision coverage

| Decision kind | Rule |
| --- | --- |
| priority | Card-specific casts, lands, outlets, mana sacrifices, then pass |
| mulligan | Land count, colored access, early plays; keep at four cards |
| choose_card | Search colors; bottom against retained hand; discard/put/dig by card value |
| target | Source-specific rules using permanent/stack/player references |
| pay_mana | Feasible residual cost, useful removal reserve, source/color opportunity cost |
| sacrifice | Cheapest current valid fodder, including draw and growth effects |
| choose_x | Largest affordable X preserving the concrete reserve |
| yes_no | Spellbomb/ward payments; known explore card; generic no-shuffle |
| choose_mode | Known scry/surveil card and Deem Inferior owner choice |
| order | Highest-value known cards first, low-value cards to bottom |
| order_triggers | Deterministic growth/draw ordering, then current index |
| exile_from_graveyard | Lowest-value available card; optional decline valued at zero |
| declare_attacker / declare_blocker | Group combat rules described above |
| assign_damage / assign_damage_amount | Complete scoring / suffix dynamic programming |

Unsupported action semantics or mixed/malformed decisions raise a diagnostic
error. Equal scores always choose the lowest current action index. The generic
rules cover reachable incidental decisions; they do not claim support for
sideboarding or playing arbitrary unrelated decklists.

## Development fixtures and review

`tests/test_benchmark_jund.py` is registered for both engines. Fixtures are
synthetic, explicitly configured and reviewed against their recorded docstring
rationale and executable outcome assertions. They are not human demonstrations
or independently accepted release puzzles. Full lines go through target choices,
payments, responses and resolution; setup-only assertions are reserved for
information contracts and pure arithmetic.

Critical lines cover colored-payment traps, tapped land/affinity sequencing,
exhausted basics, distinct Spawn payment/fodder, ward, redundant removal,
profitable/unprofitable sweeps, stacked Shaman activations, Hydra modes, Munitions
and Spawn lethal, lifelink survival, disappeared blockers, gangs and large damage
allocation. Additional checks cover hidden completions, labels, duplicate object
identity, reset registration, deterministic ties and all handler methods.
The slow scheduled-trace check runs both pilots against both opponents on Python
and native, comparing sampled/greedy decisions, visible frames and outcomes.

These checks establish the declared tactical behaviors and legal execution.
They do not establish expert-level play, robustness to every possible response,
or statistical strength. Combat grouping, utility weights and known-card baiting
remain heuristic; there is no opponent hand inference, game-tree search or
sideboard strategy.

## Reproducing the comparison

After `make setup`, commit runtime sources, then run:

```sh
OMP_NUM_THREADS=1 .venv/bin/python tools/benchmark_jund.py \
  --smoke --workers 4 --out .context/jund/smoke.json
OMP_NUM_THREADS=1 .venv/bin/python tools/benchmark_jund.py \
  --workers 4 --out .context/jund/comparison.json
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_benchmark_jund.py \
  -k scheduled_modes_and_engine_trace_parity
```

The tool uses PR64's episode scheduler, adapters and process runner. The default
legacy source freeze is PR64 `ac7e95ad583122e8031f051d0ffcac515540d9b2`; bot and
deck sources must match that commit. Both pilots use the same corrected engine
and identical fixed legacy Jund/Blue opponents. Runtime/tool source must be
committed and the installed native build must match checkout sources.

The smoke panel has 40 blocks per opponent, four seat/start slots per block:
320 specialist games. The paired panel has 400 blocks per opponent, four slots,
two opponents and two pilots: 6,400 games. It uses `benchmark-v1/dev`, disabled
automatic actions, 100 turns and a 10,000-decision cap. Wins/draws/losses score
1/0.5/0. Equal-weight differences average four-slot block differences within each
opponent cell, then average the two cells. The 10,000 percentile bootstrap draws
resample whole four-game blocks and share resampled indices across cells.
Missing, duplicate, mismatched or technical-error rows suppress inference.
A stronger-play claim requires overall lower 95% bound above zero and no cell
whose upper bound is below -0.03, per PR62.

`JundDevelopmentComparison` is a separate versioned development report because
the foundation release-result schema describes checkpoint candidates. Reports
contain raw rows, all seeds/slots, source/deck/card/parameter hashes, timing,
latency and technical failures. Scripted mode equivalence is checked on matched
traces instead of duplicating the deterministic 6,400-game panel. Machine and
concurrency must accompany latency; these are end-to-end adapter/choice timings.
The legacy adapter is explicitly diagnostic, outside the specialist's information
contract. A same-panel improvement does not establish strength against other
agents or on a reserved final split.

Measured campaign results are recorded in `benchmark-jund-results.md` after the
final source freeze.
