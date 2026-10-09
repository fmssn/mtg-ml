# Critical review: the path to a generalized Pauper player

**Astra · 8 October 2026 · evidence cutoff: 14:32 UTC.** This is an independent
review and a proposed research programme, not a report of newly completed training.
The inspected checkout is `28105450d1d448306e43f3c3056b59ff889d6b7c`.
Live-run measurements below are a snapshot at **14:11 UTC**; they are not final results.

## Executive assessment

**A strong, transferable player for a growing, explicitly supported Pauper card
pool is a credible research objective. This project has useful infrastructure
and evidence of learning, but has not yet demonstrated that capability.** More
self-play on the current six fixed lists cannot by itself establish
generalization. The largest immediate risks are a mismatch between the intended
information setting and the training inputs, incomplete legal-action coverage,
narrow evaluation, and conclusions drawn from one training lineage.

**Trustworthy card-choice research is a harder, separate objective. It is not
currently justified.** A player can beat existing bots while systematically
misvaluing cards that require strategies it has not learned. Matching tournament
matchup rates would be encouraging external evidence, but would not identify
card effects or establish that both sides are proficient. Small card-swap effects
require controlled interventions, uncertainty over training as well as games,
and sensitivity to player, opponent, sideboard and metagame adaptation.

My recommendation is to **keep the simulator-backed, masked recurrent RL
foundation; change the information contract and evaluation programme immediately;
revisit representation, discounting, population diversity and search with fair
controls; defer a wholesale algorithm replacement and large distributed runs**.
PPO is a reasonable baseline, not an established answer. The proposed direction
earns further investment only if the gates in this report pass.

The most consequential findings are:

1. **Demonstrated:** both engines omit a legal immediate win in an eleven-blocker
   trample position. Their agreement validates the port, not the Magic rules.
2. **Demonstrated:** feature set 3 onward supplies opponent archetype metadata;
   default seat conventions also encode deck identity. This differs from the
   agreed hidden-list gameplay setting. Own registered deck composition is not
   represented generally enough to condition on arbitrary lists.
3. **Demonstrated:** the 64-entity limit can remove all hand entities and action
   pointers on a crowded board. Set-6 simulation previews disappear above 32
   options. The frequency and strength impact of these cases are **unresolved**.
4. **Demonstrated:** training improves performance in supported settings. The
   fs4 fine-tune is promising. Neither its single-lineage result nor current R6
   bot/ladder scores demonstrate unseen-list or unseen-card competence.
5. **Plausible:** structured effect/deck conditioning, explicit relations and
   targeted strategic teaching will improve transfer more cheaply than simply
   increasing width. **Unresolved:** whether a single shared policy can maintain
   strong proficiency across the intended metagame within the available compute.

“Generalized Pauper” should mean: strong play across supported archetypes,
robustness to realistic list variation, and efficient adaptation to newly
implemented cards. It does **not** mean that a language embedding implements a
card's rules, or that a policy can play arbitrary future mechanics without a
correct simulator. New cards still require rules support and validation.

Throughout, **D** means directly demonstrated by inspected code, a reproduced
fixture or a recorded result; **H** means a plausible hypothesis; **U** means
unresolved. A recorded result establishes performance in that experiment, not
the causal explanation offered for it. Recommended numerical gates are project
decisions proposed here, not constants established by the literature.

## 1. What the project actually contains

### 1.1 Implemented capabilities versus intended capabilities

The [feature documentation](features.md), [representation plan](representation-plan.md),
[experiment ledger](experiments/ledger.md), [ladder protocol](experiments/ladder.md),
[match implementation](../mtg_ml/match.py), and [training code](../mtg_ml/rl/train.py)
were reconciled with archives, active metrics and GitHub PR state. Some roadmap
statements are historical: copying, hand entities, card shapes, and both set-6
preview streams are already implemented. A general relation-table representation
and empirical transfer validation remain distinct work.

| Capability | Status at the inspected code | What remains unproved |
|---|---|---|
| Python engine and native Rust port | Implemented, differential tests, scenario tests, Oracle-backed card specifications | Independent completeness and rules correctness |
| Six Pauper archetypes | Jund Wildfire, Mono Blue Terror, Red Madness, Grixis Affinity, Elves, Tron; 128 registered cards including 8 token types; 117 distinct cards in the registered mains and sideboards | Coverage of the Pauper metagame or future cards |
| Matchups | Fifteen non-mirror pairs, with fixed deck-seat assignments | Mirrors, arbitrary list distributions, unseen pairings |
| Shared player | Entity/action scoring, GRU memory, masked PPO, historical opponent pool | Robust proficiency across independently trained opponents |
| Features 4–6 | Combat/context improvements, coarse card shapes and own-hand entities, simulated and assume-pass previews | Meaningful zero-shot semantic transfer |
| Sideboarding | Real sideboards and fixed plan matrix; an interface for policies exists | Learned sideboard choice, hidden-archetype inference, adaptation within matches |
| Match play | Best-of-three-style evaluation machinery exists | R6 disables BO3 evaluation; training samples single pre/postboard games |
| Expert data | [Merged pilot](expert-pilot.md): one reviewed real opening, two action labels from a nine-minute clip | A sizeable reliable teaching set or learning benefit |
| Attention/scaling | Attention exists; draft [#53](https://github.com/fmssn/mtg-ml/pull/53) adds stacked inference, residency, BF16 and distributed PPO | Its proposed end-to-end speed and strength gains |

R6's 20% postboard fraction samples **fixed postboard decks**, not sideboard
decisions. Calling this “learning sideboarding” would overstate the implementation.
The [match code](../mtg_ml/match.py) supplies the opponent deck name to sideboard
policies, always chooses to play after a loss, and stops after three games even
if draws prevent two wins. These conventions must be explicit in the research
estimand. They are not evidence of general match strategy.

### 1.2 Results and provenance

Archived runs are under `~/mtg-ml-checkpoints/` on `h100-private`. These records
are append-only. Archive IDs below identify final artifacts; active R6 rows
identify a **metric checkpoint by iteration/game count**, not the moving
`latest.pt`. Do not replay an old score using today's `latest.pt`.

| Evidence | Code / checkpoint provenance | Recorded result | Interpretation |
|---|---|---|---|
| Overnight learning and R1 | `20261006-overnight-selfplay`; R1 children from the 8.4M parent, code `f953f1e`; flags in [ledger](experiments/ledger.md#20261007-r1--four-fine-tunes-from-the-overnight-checkpoint) and archive `launch.txt` | R1 lr-anneal: 72.1% sampled against Blue bot, L1 147; control: 65.9%, L1 103 | D: continuing training and schedule matter in this lineage. U: independent-seed reproducibility |
| R3 attention | `20261007-r3-attn`, 9.4M parent + 1M; `bdad997` plus RNG-restore fix, feature set 2 | H2H 49.2%; L1 152 versus control 146; roughly 10× slower | D: this configuration did not earn its cost. U: architecture's intrinsic value |
| R4 three-deck mix | `20261007-r4-{mix,control}`; PR #28 atop `1cf8fff`, feature set 3; final archived weights, reported metrics at 20.0M / 17.5M | Jund–Blue bot: 73.4% mix, 78.2% control | D: regression under the tested schedules. H: shared-policy interference contributes |
| R5 width | `20261007-r5-mix-{h128,h256,h256-lr3e-4}`; PR #28 code | h256: 46.3% at 2.5M versus h128 49.5%; high-LR h256 38.1% | U: width, owing to early stopping, optimization and compute differences |
| fs4 fine-tune | `20261007-fs4-ft/final.pt`, code `759358e`; parent R4 control at 17.61M; same-run fs3 continuation control | 2,000-game sampled/greedy: 78.8% / 80.3%; H2H fs3 control 52.9%; H2H parent 51.5%; L1 212.5 ± 13.8 | D: promising local improvement; recorded H2H intervals respectively 50.8–55.1% and 49.4–53.7%. U: across-training-seed benefit |
| fs4 observation audit | Same fs4 code/weights, 200 greedy games, seed 0 | 56,675 decisions; zero detected blind spots; 11 identical-option groups | D: no flagged cases in this sample under that detector. Not a proof of sufficient observations |
| R6 h128 | Code `30d9ef640d59a347e627d2ef371184b5b0801415`, feature set 6, seed 6 | At 14:11: 11,182,080 training games; latest full evaluation at 10,000,384: 63.8% / 68.1% bot, L1 63.4 | An unfinished six-deck result; compare only matched checkpoints |
| R6 h128 attention | Same deployed code, features and seed | 2,129,920 trained; latest evaluation at 1,001,472: 40.6% / 59.8%, L1 −165.9 | Evaluation is stale; not performance at 2.13M |
| R6 h256 | Same deployed code, features and seed; different LR schedule | 6,336,512 trained; latest evaluation at 5,001,216: 47.5% / 63.2%, L1 −28.0 | Different experience and optimization from h128 |

The fs4 final checkpoint's recorded SHA-256 is
`a560f282824e6a5ec357b9b677a69840029d2571571896785ea5d3d5d220a198`.
The fs4 ledger update remains draft [#46](https://github.com/fmssn/mtg-ml/pull/46),
head `b281dac6a96965c213856562cc9b78ab830e3236`; its archive was read directly.
For older rows the table deliberately preserves the available code provenance
rather than inventing a precise integrated SHA. Resolve archive `launch.txt`,
`SHA256` and feature metadata before any re-evaluation. Missing provenance is a
limitation of the comparison, not permission to guess.

R6 runs use one H100 each, 19 rollout workers each and overlapping pressure on
the same 64 logical CPUs. All use seed 6: **three architectures are not three
independent training seeds**. They also precede the later mana-sacrifice fix
merged through [#50](https://github.com/fmssn/mtg-ml/pull/50). Historical scores
must retain their engine version; future comparisons should re-evaluate all
arms on one corrected engine and separately quantify the version effect.

Draft #53's head at review is `e8662a511173a14b11a32dec7225ad6cfd513cda`.
Its [implementation and measurements](https://github.com/fmssn/mtg-ml/blob/e8662a511173a14b11a32dec7225ad6cfd513cda/docs/training-256-attention.md)
include useful correctness checks and preliminary stage timings. They expressly
do not establish the proposed 3–5× total speedup. Its frozen h256 parent is
`pool/iter_02440.pt`, 4,997,120 games, SHA-256
`0dd482bcd68e846cc7221a894b2fef711303cc183b1f4230666a5098930fbea7`.
That is not exactly the 5,001,216-game metric row above.

The card-research “trust test” is still a proposal in draft
[#44](https://github.com/fmssn/mtg-ml/pull/44), head
`d32ea12e92b065ccd9e9aaa8356856f7ed793baf`. UI and review-controller PRs do not
constitute new gameplay evidence. Draft #55, observed during this review,
adds card support for expert-pilot games; it is outside the inspected card count.

## 2. What the literature supports—and what it does not

### 2.1 Direct MTG and collectible-card evidence

**[MTG-Causal-RL](https://arxiv.org/abs/2605.06066), §6 and appendices.** This is
a useful small MTG benchmark: 56 cards, five archetypes, masked actions,
structured diagnostics and fixed opponents. Its planned protocol specifies
seven paired seeds and 300 episodes per opponent; the submitted headline and
transfer results actually use **two seeds and 30 episodes per opponent**. PPO
versus the causal variant is mixed across decks: Mono-Red 67.8% versus 70.6%,
while control and ramp remain weak. This supports testing structured inputs and
interventions, not a claim that causal PPO solves generalized Magic. Tiny-seed
bootstrap significance cannot replace independent training replications. There
is no human-strength validation. Transfer to an unseen opponent archetype is
also different from playing a new own archetype. The paper's mulligan
description warrants checking against its implementation; this review does not
declare its simulator incorrect from prose alone.

**[Generalized MTG representations](https://arxiv.org/abs/2407.05879), §§V–VI.**
This is human **draft-choice prediction**, not gameplay. Its approximately 55%
unseen-set choice accuracy is evidence that card text and structured attributes
carry transferable information. It is not a win rate, strategic mastery result,
or causal card valuation. Draft picks also reflect player preferences and the
available pack. Transfer the representation hypothesis; test it using gameplay
and held-out mechanics rather than importing the headline as gameplay evidence.

**[Hearthstone, Xiao et al.](https://arxiv.org/abs/2303.05197), experiment setup
and human evaluation.** A strong positive precedent for RL in a commercial CCG,
but the environment is a modified Hearthbreaker covering the April 2015
Blackrock Mountain era, about 350 cards and three heroes. The best non-cheating
setup uses separate models per hero. Reported training used **24 V100s and
5,856 CPU cores**, with a 23-day checkpoint. Human testing is against one
formerly top-ten streamer: two BO5 wins of 3–0 for the non-cheating model;
deck-peeking-model matches are separate. The paper supports gamma 1, reducing
policy lag, and specialization as hypotheses. It does not establish one small
shared player for unrestricted current Hearthstone, let alone Pauper.

**[Cardsformer](https://ebooks.iospress.nl/pdf/doi/10.3233/FAIA230581), §§5–6.**
MPNet card embeddings, entity transformers and an offline transition predictor
improve policy learning on 20 predefined training decks. Mulligans are skipped.
The policy uses 100M frames, about 11 hours on eight Titan XPs and 64 CPU cores;
this excludes a separately trained prediction model. Crucially, unseen cards
are restricted to relatively simple effects: unique triggers are an admitted
failure. Evaluation standard deviations from blocks of 100 games do not measure
independent-training variance. The result motivates semantic encoders and
auxiliary transition targets; it does not solve novel-mechanic transfer. The
[official implementation](https://github.com/WannianXia/Cardsformer/tree/6a0cba16298c5262ee204eeda97409064cc683b2)
was inspected: `Algo/dmc.py` regresses action values onto collected targets,
and the repository separates prediction and policy training. This is not a PPO
result or evidence that online language-model reasoning is necessary.

**[AAMAS 2025 GeneralizableHS](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2795.pdf).**
A three-page extended abstract combines T5-base card/deck representations,
generated deck-strategy summaries and a latent transition loss. Five test decks
contain up to 50% unseen cards; reported comparisons are against tree-search
baselines, Cardsformer and a prompted LLM. It gives insufficient seed, compute
and uncertainty detail for a strong scaling prediction. “Unseen” must be
qualified by stage: policy-unseen is not necessarily language-pretraining-unseen.
The [released agent](https://github.com/WannianXia/GeneralizableHS/blob/d280e503c00df27d0f396af5e57c1bbbc3233a17/GeneralizableHS/Model/Agent.py)
explicitly consumes deck-strategy embeddings and has a `no_deck` switch.
That provides a concrete ablation idea. Own-deck conditioning transfers; using
an undisclosed opponent's true deck would violate this project's target setting.

**[ByteRL exploitability](https://arxiv.org/abs/2404.16689), §§4–6.** The actual
exploitation experiments are on **Legends of Code and Magic 1.5**, not
Hearthstone. Behaviour cloning followed by PPO learns responses to a strong
fixed agent on generated deck pools. Table 1 reports 90.4% at 32 pools and
54.2% at 1,024 with BC pretraining, averaged over five runs; dependence on
ByteRL's deck construction and reduced deck diversity are explicit limitations.
The portable lesson is to train adversaries and vary their initialization.
These are discovered vulnerabilities under a protocol, not an exact global
exploitability computation for a commercial card game.

The existing [literature note](research-mtg-pauper-rl.md) is a valuable starting
point, but several claims need narrowing. MTG's worst-case computational
hardness does not imply that model-free RL is the best method for this bounded
card pool. A Gumbel policy-improvement theorem does not automatically apply to
approximate hidden-information search. A privileged critic is not a proven
“cheap win.” And “the engine is the hard part” underestimates exploration,
multiagent nonstationarity and evaluation. None of these qualifications argue
against continuing; they change what evidence continuing should produce.

### 2.2 Methods worth transferring

Implementation cost below means integration/verification burden relative to
this codebase, not a promised number of engineering weeks. The inexpensive
tests are specified more fully in §7. A method's convergence result applies
under its assumptions, not automatically to a finite neural-network experiment.

| Method and primary source | What transfers | Assumptions or evidence that do not transfer | Cost and cheapest decisive test |
|---|---|---|---|
| [PPO](https://arxiv.org/abs/1707.06347), [action masking](https://arxiv.org/abs/2006.14171) | Stable baseline with variable legal candidates; inspect probability ratios and masks | Single-agent empirical success is not convergence in self-play; a mask cannot restore omitted legal moves | Low: freeze PPO baseline; verify returns/recurrent replay; change one objective parameter at a time (O) |
| Recurrent information-state policy; [Suphx](https://arxiv.org/abs/2003.13590) | Remember revealed cards and strategic context; supervised warm starts and controlled adaptation are practical | Mahjong's domain-specific representations, scoring and oracle guidance are not a ready-made Magic solution | Low–medium: ordered history versus current GRU/event bag on reveal/forgetting scenarios, then held-out play (R) |
| [Relational deep RL](https://arxiv.org/abs/1806.01830v2) | Shared entity interactions can generalize across object combinations and counts | Box-World and StarCraft mini-games have different observation/control problems; the paper reports variable transfer and 10B-step StarCraft training, not a cheap universal gain | Medium: explicit block/target/attachment edges plus one attention layer, compared to the same encoder without it (R) |
| Explicit beliefs | Predict hidden-card/deck distributions from public evidence, with a known training prior | Finite list priors can be misspecified; a calibrated deck classifier is not necessarily a stronger player | Medium: belief auxiliary head versus same-capacity non-belief head, public-only inputs and proper held-out scoring (R) |
| [PerfectDou](https://arxiv.org/abs/2203.16406); [unbiased asymmetric critics](https://arxiv.org/abs/2105.11674) | Privileged training information may reduce variance while execution stays information-legal | DouDizhu scoring/roles differ; a state-only critic can be biased for a history-dependent actor. PerfectDou used 880 CPU cores and eight GPUs | Medium: history-plus-state critic versus parameter-matched history-only critic (O); verify actor independence |
| [NFSP](https://arxiv.org/abs/1603.01121) | Learn a historical average policy as well as a response; address cycling | Approximate neural best responses, replay coverage and perfect recall remain requirements; Leduc/limit-poker evidence is not Pauper evidence | High as a replacement; first test policy averaging/distillation against snapshots in small solved subgames |
| [PSRO](https://arxiv.org/abs/1711.00832) | Build a diverse response population and measure the empirical payoff matrix | Exact-double-oracle convergence does not certify incomplete neural response searches; payoff estimation grows quadratically | Medium for a small population: three independent lineages, specialist responses, then uniform versus empirical meta-mixture (X) |
| [R-NaD / DeepNash](https://arxiv.org/abs/2206.15378) | Search-free equilibrium-oriented dynamics are credible for long hidden-information games | Stratego has fixed mechanics and board structure; DeepNash's scale (768 learner and 256 actor TPU nodes) is not a local baseline. Theory does not certify neural exploitability | High: tiny Magic subgames with exact best responses before any full-game replacement |
| [ReBeL](https://arxiv.org/abs/2007.13544) | Search should reason about public beliefs and counterfactual values | Beliefs over infostates grow prohibitively with unknown decks, hands and histories; the paper explicitly identifies this scaling limit | Very high: a bounded public-belief combat/response subgame; compare to exact solution and PPO, not a full port |
| [Student of Games](https://arxiv.org/abs/2112.03178) | Growing-tree CFR and sound self-play unite search and learning | Generality is of an algorithm trained per game, not one universally trained network; soundness requires adequate search/value approximation | Very high: use the same tiny subgame to establish whether strategic randomization matters enough to justify it |
| Determinization / [information-set MCTS](https://doi.org/10.1109/TCIAIG.2012.2200894) | Cheap tactical search over plausible hidden worlds | Strategy fusion, incompatible future knowledge and biased world sampling; averaging omniscient plans is not a legal contingent strategy | Medium: corrected forks, constrained worlds and information-consistent decisions on 100 reviewed scenarios (T) |
| [Expert Iteration](https://arxiv.org/abs/1705.08439), [Gumbel planning](https://openreview.net/forum?id=bERaNdoegnO) | Separate discovery of good lines from amortizing them into a fast policy | Hex/perfect-information planning and accurate action values differ from noisy determinized Magic search | Medium–high: require search to beat the current policy on held-out tactical outcomes before distilling (T) |
| [DAgger](https://arxiv.org/abs/1011.0686), demonstrations and curricula | Label states the learner actually reaches; introduce rare tactical opportunities | Demonstrations can be suboptimal or use hindsight; the usual expert-oracle assumption is weak here | Medium plus annotation: reviewed scenario resets and small BC warm start, each with matched PPO control (T) |
| [DouZero](https://arxiv.org/abs/2106.06135) / Monte Carlo action-value learning | Candidate-action scoring and terminal-return learning are useful alternative controls | Team roles and game scoring differ; lower rollout bias may bring much higher variance | Medium–high: defer until objective/critic diagnostics identify PPO-specific failure |
| [IMPALA](https://arxiv.org/abs/1802.01561) | Quantify actor–learner lag; explicit off-policy correction is relevant when decoupling actors | PPO clipping is not V-trace; more throughput may reduce learning per sample | Low for diagnostics, high for replacement: lag 0/1/2 with identical batch, pool and compute accounting (O) |
| [Procgen](https://arxiv.org/abs/1912.01588), [RL statistical reliability](https://arxiv.org/abs/2108.13264) | Train/test distributions, held-out content, run-level uncertainty and performance profiles | Game-seed variation alone is not list/mechanic variation; bootstrap cannot create missing independent runs | Low: preregister transfer splits and hierarchical paired analysis (G, X) |

Two implementation checks sharpen this table. The inspected
[OpenSpiel NFSP](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/pytorch/nfsp.py)
uses a separate response learner, supervised average-policy learning and
reservoir sampling; sampling old opponents in this project is **not NFSP**.
The [ReBeL release](https://github.com/facebookresearch/rebel) covers Liar's
Dice, not the full poker system. The paper's full poker setup used 90 eight-V100
machines for data generation. “Use ReBeL” therefore describes a substantial
research and systems project, not a small algorithm switch.

The equilibrium-method evidence also deserves careful reading. NFSP's limit
Hold'em evaluation used 25,000 hands per checkpoint, but its final reported
greedy-average policy still lost to each of the three leading comparison bots.
Its stronger convergence evidence is on small Leduc, where exact exploitability
is computable. PSRO measures independently trained policy cross-play and
NashConv in smaller games; it motivates our population tests without certifying
an incomplete Pauper population. Student of Games evaluates exact small-game
exploitability for 50 constructed search-policy seeds from a **single training
run**; those are not 50 independent training replications. Its chess/Go
comparison uses very large TPU resources and up to 60,000 evaluation search
simulations. These papers establish substantive algorithmic possibilities, but
none removes the need for local compute accounting and independent roots.

For relational encoders, the inspected Zambaldi et al. preprint demonstrates
stronger compositional performance in Box-World, yet explicitly reports high
variability when transferring StarCraft resource collection from two to five
units. That distinction matters: attending across entities is an architectural
prior, not proof that a model will infer Magic's exact rules or transfer to new
effects. Supply correct relations, retain an efficient non-attention control,
and test the intended distribution shift.

## 3. Simulator and information foundations

### 3.1 A concrete missing legal win

**D:** In both Python and native engines, a 20-power Nyxborn Hydra with trample,
blocked by eleven 0/1 Eldrazi Spawn while the defender has nine life, receives
1,023 assignment options. None puts any damage on the defender or on blocker
eleven. The legal immediate win—one damage to each blocker and nine to the
defender—is absent.

The cause is [the damage-assignment cap](../mtg_ml/engine/game.py):
`MAX_DAMAGE_SPLITS = 1024`, `LETHAL_SPLIT_BLOCKERS = 10`. Beyond full enumeration,
the approximation selects lethal subsets only among the first ten blockers;
the trample-to-player branch requires selecting **all** blockers. Eleven makes
that branch unreachable. Even below eleven, retaining lethal subsets is an
action abstraction, not full legal assignment coverage.

The independent reference is the
[Comprehensive Rules effective 25 September 2026](https://media.wizards.com/2026/downloads/MagicCompRules%2020260925.txt),
**510.1c and 702.19b**: assignments may be divided among blockers; trample may
carry excess to the defender after lethal assignment to all blockers. The code's
702.19c comment is an outdated rule-number reference, separate from the actual
coverage defect. See Appendix A for a reproduction.

**Change:** use a sequential integer-allocation decision with legality-preserving
masks, or another exact compact representation. Include all blockers and
nonlethal divisions; prove completeness on exhaustively enumerable small cases.
Adding the one missing trample option repairs this fixture but not the general
coverage problem. Audit whether action decomposition, automatic mana/pass
choices, X bounds and target pruning similarly remove strategically distinct
legal lines. Do not assume all such mechanisms are wrong; distinguish proven
equivalence reductions from convenience approximations.

**U:** frequency in current self-play, or contribution to any benchmark score.
This review did not measure either. A rare forced-win omission still matters
for full-rules claims and can matter disproportionately for an archetype.

### 3.2 Why differential testing is necessary and insufficient

Keep the two-engine invariant, Oracle checks, golden digests and fuzzing.
Their combination is a substantial asset. However, a shared misunderstanding
can be faithfully ported and frozen into a golden test. Independent expected
outcomes should come from current rules/Oracle text, judge-reviewed fixtures,
and selected comparisons to another mature implementation such as
[XMage](https://github.com/magefree/mage) or [Forge](https://github.com/Card-Forge/forge).
An external engine is corroboration, not an unquestionable oracle. Compare
normalized legal actions and public successors, not internal object IDs.

Build a risk-weighted rules suite: priority and mana abilities while paying
costs; replacement/prevention effects; state-based actions; simultaneous
triggers; targeting/ward; alternative costs; graveyard/reveal visibility;
combat assignments; extra costs and sacrifice; loops and game termination.
Every supported card needs ordinary, interacting and boundary fixtures. Cards
that share an operation need cross-card tests as well as same-engine parity.

The current 100-turn cutoff produces a draw. That is an engineering limit,
not a Magic adjudication rule. Log cutoff draws separately from rules draws;
run paired sensitivity at 100/200/400 turns, and add bounded decision/loop
diagnostics. Do not silently award a loss or teach a slow control deck that
reaching the cap is its real objective. Match draw handling and play/draw
choice need a separately specified tournament or research protocol.

### 3.3 The hidden-list contract must be enforced end to end

**D:** [`state_features`](../mtg_ml/encode.py) reads the opponent's
`game.deck_names` and emits `opp:deck:*` from set 3. Changing only that metadata
changes the actor input in an otherwise identical scenario. Default deck names
may be `None`, but [`match.deck_names`](../mtg_ml/match.py) and `seat:*` then
encode the historic Jund/Blue seat convention. With fixed archetype lists,
archetype disclosure practically discloses the list.

For ordinary play, the actor may know its **own registered main/sideboard and
its own changes**, public observations, its private cards and previously learned
information. It may infer the opponent's archetype from play. It must not receive
the actual opponent list/archetype, hidden hand, library order, private
sideboard choices, simulator seed, or a preview that indirectly reveals them.
Carry legitimate information between games under the selected match protocol.

Remove explicit disclosure **and randomize deck-seat association** for new
training. Merely deleting a token from an old checkpoint is an out-of-distribution
stress test, not a fair estimate of a properly trained hidden-list player.
Maintain a separately named open-list mode for controlled research. Every
result must say which mode was used.

Counterfactual invariance tests should hold a player's full legal information
history fixed, vary compatible hidden worlds/metadata, and compare every actor
input: observations, legal candidates, previews, events and recurrent state.
Identical actor inputs must give identical logits for a fixed execution mode.
Stop comparisons when the histories actually diverge through a public reveal.
A training-only privileged critic is allowed only behind a tested separation.

### 3.4 Features can distinguish cards without describing their meaning

**D:** feature spaces use CRC32 hashing into `2^16` state and `2^15` option
bins, with set deduplication within several feature bags. Counts are supplied
through selected explicit/thermometer tokens rather than preserved uniformly.
Hash collisions are possible; saturation and bagging can also collapse relevant
quantities or relations without a hash collision. Measure distinct token
collisions separately from complete option indistinguishability.

**D:** Brainstorm and Ponder have the identical four-token `shape`:
`e:mv>=1`, `e:color:U`, `e:spell:op:custom`, `e:op:custom`.
Their names and printed types still differ, so they are **not identical full
network inputs**. The finding is about missing effect semantics, particularly
for unseen cards. A bag of operation names also loses which cost, target,
condition and magnitude belong to which ability, and in what order effects
execute.

**Change:** progressively encode structured effect trees: typed operations,
arguments, targets, conditions, zones, timing and linked costs/abilities. Keep
exact numeric channels where thresholds matter. Replace opaque custom-effect
shapes with verified structured semantics as cards permit; report uncovered
custom effects instead of representing all as semantically equivalent.
Represent the own deck as a count-preserving multiset and known zone changes,
not an archetype label alone. Distinguish the registered list from uncertain
remaining-library composition after unknown mill, shuffle or hidden movement.

**H:** a hybrid of structured semantics plus cached frozen Oracle-text embeddings
will transfer better than names alone. Test structured-only first. Language can
help with shared concepts and unfamiliar descriptions; it can also memorize
familiar card names, blur exact rules and encode external strategic priors.
Use name masking, meaning-preserving renamed aliases and mechanic holdouts to
separate those possibilities. Cache embeddings once per card; there is no
current evidence requiring an LLM in the action-selection loop.

**D:** entities are battlefield first, then stack, then own hand, capped at 64.
A fixture with 65 permanents loses the Lightning Bolt hand entity and its
pointer. Global hand-name/count features still exist. This specifically damages
the richer representation and action/entity links. Reordering otherwise
equivalent objects can change which information survives.

Use ragged/bucketed entities, or a rigorously justified compression of truly
exchangeable objects with counts and action mappings. Reserving hand slots is a
useful mitigation but merely moves the information loss. Report overflow rates
by archetype and board size, and test permutation equivariance with pointers
remapped correctly.

**D:** set-6 previews skip decisions above 32 options and stop at boundaries
including hidden transitions. The `simp` stream assumes the opponent passes.
It is a conditional tactical feature, not the expected result of an action.
The eleven-blocker fixture skips both streams. **H:** these cutoffs remove help
precisely when interactions are difficult; this needs prevalence and ablation
evidence. Keep separate missing/unknown/assume-pass tags and test with previews
removed, direct features only, and semantically encoded cards.

Finally, the GRU receives mean-pooled event bags with a 256-token limit between
own decisions. It can remember observations across decisions, but cannot
recover event order that was removed before input. Audit revealed-card identity,
public ordering and per-object relations; benchmark ordered event encoding on
tasks requiring recall. “The policy has a GRU” does not establish sufficient
memory or learned beliefs.

## 4. Learning, self-play and earlier conclusions

### 4.1 PPO and GAE: sound core, unsettled objective

The inspected [PPO loss](../mtg_ml/rl/ppo.py) uses the standard clipped ratio,
masked categorical entropy and whole-trajectory recurrent replay from zero
under current weights. [`_finish`](../mtg_ml/rl/rollout.py) computes terminal
rewards, bootstrapped TD residuals and GAE in reverse order. These are positive
design choices. Existing focused tests were run during this review; they are
implementation checks, not an independent proof of the entire trainer.

The default `gamma=.995`, `lambda=.95` operates per **recorded learner
decision**. A terminal reward 100 decisions away has gamma weight about 0.606;
at 200, about 0.367. Mana payments, priority passes and decomposed choices can
change that distance without changing the strategic turn horizon. The
gamma–lambda trace decays faster still; it relies on the critic to bridge long
plans. **D:** the mathematical objective differs from undiscounted win/draw/loss
utility. **H:** this disadvantages long control/combo lines or rewards faster
wins and delayed losses. Actual strength impact remains U.

Test gamma 1 with decision-lambda unchanged, then turn-gamma with
decision-lambda. Treat lambda as a bias/variance control, not part of the
game's utility. Preserve administrative decisions if strategically necessary;
otherwise test equivalence-preserving compression and ensure the reward/time
semantics remain coherent. Avoid equating higher explained variance with better
strategic learning when the target distribution changes.

Potential shaping uses `w * (gamma * phi_next - phi_now)` and terminal potential
zero, so its discounted telescoping form is appropriate for the chosen gamma.
However, **D:** `phi=(my_life-opponent_life)/20` is unbounded by one; a legal
100-versus-20 life state gives 4. The claimed general return bound
`±(1+abs(w))` used for value clipping is therefore unjustified while shaping is
active. At `w=.2`, a one-step loss from that state has shaped return −1.8,
outside ±1.2. The code clips bootstrap values, not that terminal reward, so
this is a bootstrap-bound problem rather than evidence that final rewards are
silently clipped. R6's sampled current shaping is zero; no contribution to its
present score is established. Revisit clipping with a genuinely bounded
potential or shaping disabled.

A larger symmetric critic failed to improve held-out Monte Carlo prediction
much in the prior offline probe. That does not test an asymmetric
**history-plus-state** critic, altered targets, or policy improvement from
lower-variance gradients. Conversely, a lower value loss alone does not justify
a privileged critic. Keep actor/critic information separate and compare real
learning curves, advantage diagnostics and hidden-information invariance.

The current pipeline has one-update policy lag; #53 proposes two. Behaviour
log-probability ratios are necessary but do not eliminate state-distribution
staleness, critic-target staleness or recurrent-policy drift. Measure KL at the
first learner update, by decision kind as well as globally, and test lag on a
fixed batch/pool. A throughput increase is useful only if it improves validated
strength per hour.

### 4.2 Self-play needs an adversarial test, not only an older-self ladder

Self-play average score is close to 50% by construction in a symmetric
population; it cannot establish absolute strength. L1 contains four snapshots
from one lineage spanning only 0–79 base Elo. It is useful as a stable local
reference. Strong new players above that range extrapolate, and a scalar rating
can hide cycles, matchup asymmetry and shared blind spots.

Use cross-play matrices across independent seeds and architectures, specialist
policies, diverse bots and budgeted response learners. Examine per-deck losses
and lower-tail performance rather than only a metagame average. Explicitly
separate adapting to one opponent from adapting to a distribution. Preserve
challenging historical opponents when new training forgets their counters.

For shared-policy interference, compare a joint model with deck-conditioned
heads/adapters and per-archetype specialists. Match **practice per task** as well
as total compute: a specialist bank consumes more total compute if each member
gets the same games as the shared model. Log per-task gradients/retention if
useful, but a conflicting-gradient diagnostic is not itself a win-rate cause.
Do not multiply to all pairings by reflex: transfer across pairings is exactly
what a shared model should earn through held-out tests.

### 4.3 Reclassifying past experiment verdicts

| Prior decision | What survives scrutiny | What should be retested |
|---|---|---|
| Reject search distillation | The recorded implementation hurt learning and was expensive | The ledger documents lost determinization below the root, an unenforced evaluation cap, and a margin gate using an unevaluated reference. Repair correctness; first test search as a tactical auditor, then as teacher |
| Reject attention | That eager-server configuration was poor value | R3 attention had fresh Adam and no copied pool while the control resumed; the inference path was roughly 10× slower. Equal initialization, pool, tuning and both games/time are required |
| Width 256 inconclusive | Appropriate verdict | The 2.6M stop and LR sensitivity do not show a capacity ceiling. Do not turn “may improve later” into permission for unbounded training |
| Reject PFSP | Near-copy opponents supplied little actionable diversity | Test genuinely diverse lineages and specialists before changing sampling weights; PFSP is a scheduler, not a diversity generator |
| Reject per-turn discount | That joint gamma/lambda change was harmful | Test gamma 1 and turn-gamma/decision-lambda independently, with equal tuning budget and terminal utility |
| Drop bigger critic | Extra symmetric capacity did not help the offline target | Information, history, target and clipping are different questions from width |
| Exploiters found “no gap” | Roughly 300k games from a related initialization found no convincing improvement | Multiple fresh and BC/warm-started adversaries, plateau controls and held-out evaluation; failure to find a response does not certify robustness |
| Mix costs about 5pp | A real benchmark regression in the tested lineage | One seed, evolving schedule and shared feature updates do not isolate interference completely, even after accounting for fewer primary-matchup games |
| Adopt fs4 | Sensible provisional engineering decision: fixes observed gaps, promising H2H | Three/five independent lineages, modern engine re-evaluation, broader opponents and transfer tests |
| High Red win rate means strong Red | Red learned to exploit an unfamiliar-to-Jund matchup | [The review](red-madness-review.md) documents weak frozen Jund, unused plot and wasted madness; 88.5% against that opponent is not general Red proficiency |

These are single-lineage experiments unless explicitly stated otherwise.
Thousands of evaluation games reduce sampling error around those policies;
they do not remove initialization/exploration uncertainty or a shared ancestor's
blind spots. Preserve the historical ledger; annotate its scope rather than
rewriting recorded outcomes as though the new interpretation were measured then.

## 5. Card-choice research needs its own validation

### 5.1 Three different questions

Let `S(D, π, O, B, w)` be expected match score for deck `D`, player `π`,
opponent policies/lists `O`, sideboard policy `B`, and metagame weights `w`.
Use win = 1, draw = 0.5, loss = 0 unless a separately defined tournament-points
objective is required. For a deck intervention `D → D′`, report:

| Estimand | Controlled comparison | What it means |
|---|---|---|
| **Fixed-player effect** | `S(D′, π, O, B, w) − S(D, π, O, B, w)` | How the swap affects this already-trained player. A unfamiliar-card penalty is part of this result |
| **Equal-adaptation effect** | Adapt separate copies of each independent parent on D and D′ with the same games, optimizer/search/tuning budget and opponent distribution | How the swap performs after a specified learning opportunity. It is not necessarily the effect under optimal play |
| **Adaptation sensitivity** | Repeat with opponent frozen/adapted, alternative reviewed sideboard plans, and preregistered metagame weights | Whether the conclusion survives strategic response and plausible deployment choices |

Do not call the second effect “true card strength.” Some variants may remain
underlearned after equal budgets. Plot the intervention effect at 0, 0.1M,
0.5M, 1M and 2M adaptation games; measure proficiency and slope on both sides.
If the sign changes with budget, the current answer is budget-sensitive. A
specialist/teacher control can reveal residual proficiency gaps without
pretending to be perfect play.

Keep main-deck and sideboard interventions distinct. An unchanged sideboard
plan can become illegal or strategically inappropriate after a main-deck swap;
predefine legal corresponding plans, then separately allow reviewed adaptation.
Match-level memory and what the opponent learned in game one matter. A game-one
score cannot substitute for the claimed BO3 outcome.

### 5.2 The tournament matrix is a useful external check

The proposed trust test asks for credible human-like matchup rates and no
obvious deck bias. Keep it as an **external consistency check**. It can expose
wrong lists, a broken mechanic or asymmetric proficiency. It is not sufficient
for card research:

- Equally weak policies can generate plausible matchup rates. Some mistakes on
  the two sides can cancel in the aggregate.
- Human records mix pilot skill, list versions, sideboarding, event selection,
  date and metagame. A card's inclusion is not randomized. Skilled pilots may
  select a card precisely in matchups where it is useful.
- An aggregate archetype matrix loses within-archetype list variation, exactly
  where small card-swap effects live.
- Tuning repeatedly until the matrix matches consumes it as development data.
  A later time/event holdout is needed for external validation.
- “No older snapshot beats it” only checks that opponent set; it is not an
  approximate best-response test or a certificate of optimality.

The local Q3 deck-fidelity audit also identifies real list mismatch: current
Jund has Hydra and extra Toxin Analysis relative to its stock reference, while
Blue has Force Spike and Plunder instead of that reference's Dispel, Deep
Analysis and Spell Pierce configuration. These are meaningful variants, not
necessarily bad lists. Their results should not be calibrated indiscriminately
to the aggregate stock archetype. The audit is at
`/Users/fabsi/repos/mtg-ml-orchestration/deck-fidelity.md`; source dates are
5 July–2 October 2026. Preserve the extracted list/event dataset and its hash in
the eventual experiment archive, since this machine-local path is not portable.

For observational human data, stratify or model pilot, event/date, exact list,
seat where known and sideboard regime; report missingness and overlap.
Propensity adjustment can reduce measured confounding but cannot eliminate
unknown pilot skill or selection effects. Human agreement is triangulation.
The simulator supplies randomized counterfactual experiments **conditional on
its rules and policies**; it does not automatically identify a causal effect
for human tournaments.

## 6. Decisions

| Design decision | Verdict | Reason / gate |
|---|---|---|
| Custom fast simulator with Python/Rust parity | **Keep** | Essential throughput and controllability; add independent rules/coverage checks |
| Legal-option masking and decomposed actions | **Keep** | Scalable policy interface, conditional on legal-action completeness |
| Capped damage-assignment approximation | **Change** | Reproduced absent winning action; require exact coverage |
| Silent entity truncation | **Change** | Removes meaningful entities/pointers; do not scale into unmeasured overflow |
| Disclosed opponent archetype in ordinary play | **Change** | Violates the selected information setting; randomize seats too |
| Fixed archetype identity as own-deck conditioning | **Change** | Use registered list counts and legal history |
| Hashed bag features | **Keep** as baseline; **revisit** as final representation | Cheap and useful; quantify collisions, semantic aliasing and numeric saturation |
| Coarse card shapes | **Change** incrementally | Preserve effect arguments/relations; custom-op names are insufficient semantics |
| GRU memory | **Keep**; **revisit** event encoder | Full-trajectory replay is valuable; pooled/truncated input loses history |
| Set-6 simulated/assume-pass previews | **Keep** provisionally | Helpful engineering prior; test coverage, cost and dependence separately |
| Entity relation attention | **Revisit** | Needs genuine relational input, fair initialization and a usable inference path |
| Width increase alone | **Revisit** | No decisive capacity test; require equal-games and equal-compute curves |
| PPO | **Keep** as reference | Working baseline; no evidence yet compelling a replacement |
| Gamma .995 per administrative decision | **Revisit** urgently | Objective depends on action decomposition; test undiscounted utility |
| Life shaping and value clipping | **Revisit** | Shaping telescopes, but the stated bound is invalid at high life totals |
| Symmetric shared critic | **Keep** as control; **revisit** privileged history/state critic | Current negative critic result answers a narrower question |
| One shared policy | **Keep** as hypothesis | Compare own-deck conditioning, adapters and specialists; require retention |
| Historical snapshot pool / PFSP | **Change** population, **revisit** weighting | Near-copies cannot test or train robustness |
| Search-assisted learning | **Revisit** after correctness and tactical gate | Prior defects invalidate a family-wide rejection |
| Reviewed demonstrations/curricula | **Revisit** with small pilots | Good candidate for rare multi-step lines; no measured learning benefit yet |
| Fixed sideboard matrix | **Keep** for controlled baseline; **change** for sideboard claims | Specify the policy and test alternative plans before attributing card effects |
| L1 and scripted benchmark | **Keep** as regression checks | Supplement with held-out distributions, cross-play, responses and experts |
| Tournament-matrix trust test | **Keep**, narrower claim | External consistency, not card-level causal validation |
| Whole-game NFSP/R-NaD/ReBeL/SoG replacement | **Defer** | First establish which failure PPO cannot resolve; validate alternatives on bounded subgames |
| Multi-machine synchronous scaling | **Defer** pending profile | Independent experiments currently buy more information; CPU supply is material |

## 7. Executable experiment programme

These specifications describe follow-on work. New harness features, fixes and
training launches have **not** been implemented or run by this review. “Exact”
here means a protocol that can be frozen into manifests before execution; where
data curation is required, its acceptance rule is specified and the campaign
must not start with that field silently missing.

### 7.1 Common contract for every experiment

**Reproducibility.** Freeze code commit and dirty-state check; Rust build and
environment lock; rules/Oracle version; deck multisets and sideboard plans;
feature version and encoder artifacts; training initialization, optimizer, RNG,
opponent-pool hashes; launch flags; GPU UUIDs/CPU affinity; outcome logs and
replays for failures. Hash final weights and every evaluated snapshot. Record
real games, learner decisions, environment decisions, gradient updates,
GPU-hours, CPU-core-hours and wall time. Count both seats' data explicitly so
that “frames” cannot drift between experiments. Keep engine-correctness changes
outside representation/algorithm A/B comparisons.

**Independent seeds.** Screening uses three complete independent lineages,
training seeds `101, 102, 103`; confirmation uses **five new lineages**,
`501…505`. Paired arms may share the same initialization within a seed, but
different seeds must have independently generated roots and opponent histories.
Three continuations from a single published parent are continuation-seed
replications only. Label them that way; they do not satisfy confirmation.
Do not transfer information from sealed card/archetype holdouts through a
pretrained parent or its opponent pool.

**Evaluation split.** Use separate deterministic streams named
`review-v1/dev`, `review-v1/final` and `review-v1/power-pilot`, generated by
SHA-256 of `(stream, cell, block_index)` and converted to the simulator's seed
range. Also separate deck/list hashes and whole video/scenario families.
Reserve final streams before tuning; run them once per frozen confirmation
claim. Use development data for checkpoint selection. Final failure remains a
failure, not a reason to choose a different checkpoint against the same seeds.

**Compute fairness.** Each learning comparison has two endpoints: at the
specified common game count and at the control's measured total resource
budget. For the latter, hold GPU allocation and CPU allocation fixed, use
checkpoint-at-budget stopping, and report actual charged GPU-hours. If an arm
uses more GPUs, its equal-compute allowance shortens accordingly; report an
additional equal-elapsed comparison separately. Include teacher/search and
pretraining cost. Equal games is sample efficiency, not equal compute.

**Opponent suite.** Freeze a development suite of all six scripted bots plus
available archived strong policies, and a final suite of independently trained
lineages, deck specialists and held-back exploiters. Use both player roles and
play/draw starts. For main comparisons use sampled play; greedy is a separate
deployment mode, not a checkpoint-selection shortcut. Report all six archetypes,
six mirrors and fifteen non-mirror pairs where applicable, preboard and
postboard separately. Uniform matchup weighting is the primary aggregate;
preregistered metagame weighting is secondary.

Historical policies trained with disclosed archetypes need explicit treatment:
either retain their original inputs and label them privileged stress opponents,
or evaluate masked-input degradation as a separate diagnostic. Neither is the
primary fair hidden-list comparison. Newly trained confirmation opponents must
obey the same information contract. Cross the list and pairing axes in the
evaluation panel: seen list/seen pair, unseen list/seen pair, seen list/unseen
pair and unseen list/unseen pair. This separates the shifts instead of
attributing every loss on the combined holdout to representation.

**Uncertainty.** Score draws as halves but also report win/draw/loss and
termination reasons. The independent evaluation unit is a seed block containing
the paired roles/starts for both arms; in BO3 it contains complete matches,
not independent games. Form each arm difference inside the block first. Use a
hierarchical paired bootstrap: resample independent training-seed pairs, then
paired blocks within task/opponent strata; keep games within a match and shared
opponent/seed blocks together. Show per-training-seed effects as well as an
aggregate interval. If claiming transfer to a distribution of lists, resample
list clusters too; if claiming only the fixed panel, keep that panel fixed.
With five seeds, intervals can still be wide: report that, rather than
pretending more evaluation games solve training variance.

For a target difference `δ`, first run 400 development seed blocks and estimate
the variance `s²_D` of paired block differences. A planning approximation for
80% power at two-sided 5% is:

`n_blocks ≈ (1.96 + 0.84)² × s²_D / δ²`.

For a 2pp effect this is about 1,960 blocks if `s²_D=.10`, or 4,900 if `.25`;
for 1pp, about 7,840 or 19,600. In a four-game block these mean roughly
7,840–19,600 games for 2pp, and 31,360–78,400 for 1pp **per comparison cell**.
They are illustrations, not measured project variances. BO3 blocks cost four
matches instead; inflate for the observed training-seed component, multiple
planned contrasts and attrition. Use simulated power from the pilot hierarchy
for final sizing. A narrow Wilson interval around one win rate is not an
interval on a paired difference, and fractional draws are not binomial trials.

For architecture screening use a preregistered 3pp worthwhile effect; for
confirmation use 2pp unless a different practical margin is justified in
advance. Use Holm adjustment within each selected family of confirmatory
contrasts. Stop only at budget, correctness failure, or a preregistered
development futility look at half budget; no repeated peeking at final p-values.
Statistical uncertainty that crosses a gate is **inconclusive**, not a pass.

### 7.2 F — Are we training the intended game? (priority 1)

| Field | Specification |
|---|---|
| Hypothesis | Engine/actions obey supported Magic rules; actor inputs depend only on legal information; representation omissions are rare enough to characterize |
| Control / treatment | Current checkout versus candidate correctness fixes, run on the same fixtures. Compare Python and native independently to an external expected outcome, not just to one another. For truncation/hash probes compare production encoder with uncapped/unhashed diagnostic output |
| Data split | Initial 120 fixtures: 20 each for combat, costs/mana, stack/priority, targeting/replacement, zone/visibility, termination. Hold back one third by interaction family. Add 10,000 generated small damage-allocation cases and 1,000 compatible hidden-world pairs. Prevalence sample: 100 games per each of 21 matchup cells, half preboard/half postboard, over bots and three archived policies where supported |
| Initialization / budget | No learning. Fresh engine per fixture; at most 2,100 sampled games plus the bounded generated checks. Three independent generation seeds; five for confirmation after a defect fix |
| Opponents | Irrelevant for independent fixtures; prevalence strata include scripted and learned policies, tagged separately |
| Primary metrics | Illegal/missing legal outcomes; actor-input/logit mismatches; overflow and preview-skip rate per 1,000 decisions by kind/deck; semantically distinct aliased options. Also termination-limit rate |
| Stop / threshold | Any information leak or missing winning line blocks claims and affected training. Require zero failures on the agreed correctness suite and small exhaustive allocations. Require zero lost action pointers in supported states. A >1% preview-skip rate in any strategic decision stratum triggers a budget/representation ablation, not an automatic rules failure. Zero observed failures is sample-limited evidence |
| Reproducibility | Save fixture, rules citation, expected action/successor, both-engine output and minimal failing replay. Print pre-hash tokens and object-to-pointer map for representation cases. Judge/adjudicator signs disputed expectations |

Run the 100/200/400-turn sensitivity on identical seeds here. Predeclare a 0.5pp
score-change margin and report the proportion of cap-induced draws. If an
interval cannot exclude a larger effect, increase the limit or retain a clear
restricted-game label. Current hidden-list and damage fixtures already fail
the intended contract; first-campaign training starts after those fixes.

### 7.3 G — Does the agent generalize? (priority 2)

This experiment is a **matrix of distinct shifts**, not one pooled holdout score.
Each comparison tests the same evaluation decks under both policies so that
deck strength is not mistaken for transfer. An in-distribution specialist on
the held-out content is an optional proficiency reference, with its extra
training cost reported; raw seen-versus-unseen matchup win rates are not enough.

| Field | Specification |
|---|---|
| Hypothesis | A hidden-list, own-deck-conditioned model transfers to realistic list changes and unseen pairings; structured semantics further improves transfer and adaptation |
| Control / treatment | C: corrected set-6-style encoder with exact own-list counts and hidden-opponent contract. S: C plus structured card-effect encoder, names retained, same h128 recurrent core and comparable parameter budget. No attention/language/search changes in this first contrast |
| Lists | Per archetype curate 12 distinct supported 75-card lists: 6 train, 2 development, 4 final. Deduplicate exact mains/sideboards and source events. Two final lists are deliberately held-back 1–2-slot perturbations of training bases (a local robustness test); two differ by 4–8 slots and belong to separate source/list families (a broader transfer test). Use dated real lists when fully supported; label curated variants separately. Preserve land count and legality unless explicitly testing mana changes. Freeze all exact counts/hashes before training |
| Pairings | From the fifteen non-mirror pairs, reserve `jund_elves` for development, `blue_tron` and `madness_affinity` for final. Train the other twelve. Ten percent of games are mirrors on training lists, uniformly across six archetypes; the rest uniformly across training non-mirror pairs. Final mirror tests use held-out lists |
| Archetypes | Separate pilot folds: Elves held out for development, Tron held out for final, each excluded from both seats, pools and demonstrations. Later rotate through all six before claiming archetype-general transfer. Do not reuse the all-six parent for this test |
| Cards | Separate folds: development excludes Mental Note, Elvish Mystic and Lightning Bolt globally; final excludes Ponder, Toxin Analysis and Crop Rotation globally. Exclude them from both players, sideboards, pools and transition/demo pretraining; curate replacement lists before training. Classify overlap in effect primitives. A later genuinely new-mechanic fold is required; these six cards alone do not establish arbitrary-mechanic transfer |
| Initialization / budget | From scratch, 3 seeds/arm at 10M games for list+pair screen. Save 0/0.5M/1M/2M/5M/10M. Run each archetype/card fold separately at the same 10M budget after the core screen passes. For transfer adapt fresh copies at 0/0.1M/0.5M/1M/2M games on the new distribution, with 20% replay from old tasks. Confirm the selected contrast with 5 new roots |
| Opponents / primary metric | Frozen diverse suite on identical held-out lists and pairings; paired score difference S−C at zero adaptation and area under the score-versus-adaptation curve (trapezoidal average over 0–2M). Report retention on old tasks and the worst archetype alongside the uniform mean |
| Stop / threshold | Screen at 5M for futility only if all 3 seeds show no transfer benefit and development's upper bound excludes +3pp. At full budget, promote with ≥3pp transfer/AUC gain and no >2pp overall in-distribution loss. Confirmation: mean benefit ≥2pp with paired 95% lower bound >0; no archetype's retention loss convincingly exceeds 3pp. Failure on one shift narrows the claim; it must not be averaged away |
| Reproducibility | Hash split manifests; assert no forbidden cards/archetypes in either seat or any auxiliary data. Record adaptation exposure by matchup and compare to equal-compute endpoints. Novelty relative to text pretraining is separately disclosed |

The exact lists require curation; their contents are an explicit preflight
artifact, not a licence for silent deck changes during training. If twelve
credible lists cannot be assembled for an archetype, report fewer and narrow
the diversity claim rather than manufacturing “real-world” variety.

### 7.4 X — Is apparent strength robust? (priority 3)

| Field | Specification |
|---|---|
| Hypothesis | Improvements survive independent opponents and do not primarily exploit one shared lineage |
| Control / treatment | Evaluate selected C/S roots plus fs4 and R6 references on the same corrected engine. Train response learners against frozen targets using (a) random initialization and (b) BC/warm start from a separate lineage. For population follow-up compare current snapshot scheduling, a fixed uniform diverse pool, then PSRO-style empirical weighting on that identical diverse pool |
| Split | Development cross-play matrix versus a separate final matrix containing held-back training roots, list variants and response seeds. Include all 21 matchup cells. Tactical evaluation: 300 judge-reviewed states, 50 per archetype, split 150 development/150 final by strategic template |
| Initialization / budget | Use G's independent roots. Initial response screen: 3 seeds × 2 initialization families × 2 player roles × 1M games = 12M games, each run training on a uniform mixture of the two final-held-out pairings. Save at 0/0.25M/0.5M/1M. If responses improve ≥1pp in the final quarter, extend to 3M. Confirmation uses 5 seeds; broaden target/matchup coverage before robustness claims |
| Opponents | Target frozen during each response run; train on development seeds and evaluate responses only on disjoint final seeds. Include a response-to-weak-bot positive control to establish the response learner can learn |
| Primary metric | Improvement achieved by the adversary over its own unadapted baseline against the fixed target, paired by role/list; complete cross-play score matrix and lower-tail performance. Tactical metric is rate of avoidable outcome-changing errors, allowing multiple acceptable lines |
| Stop / threshold | Any confirmed ≥5pp response gain, or ≥10pp deficit to a diverse opponent on a supported archetype, is a vulnerability warranting targeted training. A ≥2pp confirmed cross-play gain with no >3pp retention regression supports adoption. Before “strong play,” require ≥90% success on the agreed forced tactical suite and no catastrophic recurring class of error. No discovered exploit only means none found under the stated budget |
| Reproducibility | Preserve target/opponent hashes, response curves, best-response selection protocol and seeds. Report the target's role-specific baseline—50% is not the equilibrium value of an arbitrary asymmetric matchup |

Add a human check after this gate, not as a substitute: recruit at least three
experienced Pauper pilots covering different archetypes; preregister 60 BO3
matches for an initial feasibility panel, balanced for deck/seat and concealed
model identity. Use adjudicated legal observations and record technical failures.
This is a strength/error-discovery pilot, not a high-precision certification.
Confirm with a sample sized from between-pilot and match variance, fresh pilots
where possible, and a predeclared reference skill level. “Strong” is otherwise
too easy to redefine after seeing results.

A proposed operational human-strength gate is at least 50% mean match score
against that preregistered experienced-pilot pool, with a 95% lower bound above
45%, and no supported archetype showing a confirmed deficit greater than 10pp.
This is noninferiority to a specified opponent population, not a superhuman-play
claim. A 60-match pilot is unlikely to resolve all of those conditions; size the
confirmation accordingly and include pilot-level clustering.

### 7.5 R — Which representation earns its cost? (priority 4)

| Field | Specification |
|---|---|
| Hypothesis | Structured semantics, own-deck counts and relations improve held-out play; memory and previews contribute distinct benefits |
| Control / treatment | Reuse C/S from G. In successive matched pairs: S versus S+one relation-biased attention layer (h128, 4 heads); winner versus +cached frozen text embeddings projected to 128; winner versus name features removed. Separately compare previews {none, direct only, direct+sim, direct+sim+simp}; memory {current GRU/event bag, no memory, ordered-event GRU}. Each pair changes only the named factor |
| Split | G's list/pair/card splits plus F's permutation/aliasing fixtures. Language encoder tuning and transition pretraining obey card holdouts. Use evaluation-only input ablations as diagnostics, then train matched arms to establish causal learning effects |
| Initialization / budget | 3 fresh seeds at 5M games per selected pair; 5 new seeds at 10M to confirm one finalist. New modules initialized identically by rule, optimizer/pool fresh for both. Allow two preregistered LR choices per architecture on development only, charged equally. Do not launch the full Cartesian product |
| Opponents / metric | G/X suite; held-out score and adaptation AUC, plus examples/hour, decision latency, memory and truncation/aliasing. Require a supported bridge from semantic prediction gains to gameplay gains |
| Stop / threshold | Promote ≥3pp screen gain; confirm ≥2pp with lower bound >0, retention loss <3pp and no equal-compute regression >2pp. A slower arm may be retained as teacher only if its quality gain is demonstrated and separately costed. Stop expensive arms if the matched-compute upper bound excludes the worthwhile gain |
| Reproducibility | Save encoder/text-model versions, actual frozen embeddings, relational schemas, token collision manifests, parameter counts and all tuned controls. Renamed-card controls preserve exact mechanics and mask identity consistently in state/action/event channels |

For explicit beliefs add a separately parameter-matched auxiliary head predicting
opponent card counts/archetype posterior from public history. Evaluate log loss
and calibration on held-out lists under a stated prior, then action/strength
benefit. Do not reward beliefs for reconstructing the simulator's undisclosed
metadata. A learned latent transition predictor is another separate treatment;
measure actual consequence prediction on held-out interactions, not only latent
loss, which can improve without strategic meaning.

### 7.6 O — Is the objective or critic limiting learning? (priority 4, cheap diagnostic)

| Field | Specification |
|---|---|
| Hypothesis | Undiscounted terminal utility and/or better critic information improve long-horizon play without unacceptable variance |
| Control / treatment | From the same corrected baseline: O0 gamma .995/decision, lambda .95/decision; O1 gamma 1, lambda .95/decision; O2 gamma .97/game-turn, lambda .95/decision. Shaping off in all primary arms. Separately compare history-only critic versus equal-size history+state critic; separately lag 0/1/2 with unchanged batch/pool. Never bundle these |
| Split | Development/final long-horizon templates covering resource preservation, delayed lethal, multi-action combos and information use; also G/X full-game splits |
| Initialization / budget | Paired forks of each of 3 independent G control roots, same optimizer and frozen pool history, 2M further games; a continuation-only screen. Confirm winner from 5 new roots at 10M. First cheap run is O0/O1 only; O2/critic/lag follow if justified |
| Opponents / primary metric | Diverse frozen suite, final terminal score and long-horizon scenario success. Secondary: value calibration, advantage variance, first-update KL by decision kind and return sensitivity to equivalent administrative decomposition |
| Stop / threshold | Same 3pp/2pp screen/confirmation gates; abort numerical instability or >5pp development regression across all seeds at halfway. Lower value loss without stronger play is not adoption evidence. Any actor dependence on privileged input is a correctness failure |
| Reproducibility | Hand-computed return/GAE fixtures, terminal/truncation distinction, per-step time increments, actor/critic input hashes and schedule metadata. Charge tuning equally; report objective and learning efficiency separately |

### 7.7 T — What teaches missing strategies? (priority 5)

| Field | Specification |
|---|---|
| Hypothesis | Missing lines are partly an exploration/data problem that targeted exposure can solve more cheaply than undirected self-play |
| Control / treatment | Matched PPO continuation P. C: replace 10% of training starts with curriculum scenarios, ordinary terminal reward. D: reviewed BC warm start plus the same PPO continuation. S: corrected information-consistent search targets plus the same PPO continuation. C, D and S are separate arms; retain 90% full games and compare against a control given the same teacher/pretraining compute budget |
| Split | 300 reviewed scenarios and 1,000 accepted action/line labels; 60/20/20 train/dev/final by interaction template, card combination and source video. No adjacent frames from one game across splits. Include Shaman–Toxin, mana preservation for madness, spell sequencing, counter timing, combat and slow resource plans; allow acceptable-action sets and uncertainty labels |
| Initialization / budget | 3 independent G roots; 2M full-game-equivalent PPO games/arm, counting reset decisions separately; at most 20k BC updates or 10 GPU-hours, whichever first. For search, first 100 scenarios at 16 and 64 simulations × 8 compatible hidden worlds, ≤64 environment decisions per simulation; log all expansions and enforce caps. Search+distillation ceiling 20 GPU-hours and 400 CPU-core-hours per seed before the 2M continuation. Confirm selected arm with 5 roots |
| Opponents / primary metric | Full-game diverse suite plus final scenarios; primary forced-opportunity success and reduction in avoidable outcome-changing mistakes; full-game score is the transfer gate. Measure opportunity frequency, not just raw combo counts |
| Stop / threshold | Search must improve final tactical success by ≥10pp with paired lower bound >0, respect hidden-world invariance and beat a budget-matched policy-rollout baseline before it supplies targets. Teaching must retain ≥10pp scenario improvement with ≤2pp full-game regression; adoption for gameplay needs ≥2pp confirmed full-game benefit or a preregistered critical-error reduction with noninferiority. If gains vanish on new templates, narrow the claim |
| Reproducibility | Archive accepted/rejected annotations, legal observations, alternatives, disagreement and source timestamps. Separate narrative hindsight from what the player knew. Record search seeds, hidden-world sampling prior, evaluator versions and teacher cost |

Scenario resets must replay a legal prefix into the recurrent state or mark the
start as an explicit new information state with sufficient history. Starting a
memory-dependent position with arbitrary hidden state makes the curriculum a
different problem. Apply a DAgger-like second round only after collecting
learner-induced errors and acquiring reviewed labels; never treat a generated
commentary as a ground-truth expert action.

### 7.8 V — Can it rank card changes? (priority 6)

| Field | Specification |
|---|---|
| Hypothesis | The evaluation detects large known effects, returns zero for null interventions and yields stable small-swap rankings after equal adaptation |
| Control / treatment | Twenty nulls: unchanged deck copied/reordered, identity-preserving aliases and equivalent card-order serialization. Large controls: replace four relevant nonlands with off-colour basic lands (negative diagnostic) and reverse (positive); these are sanity controls, not realistic recommendations. Then six preregistered 1–2-slot realistic swaps, with reviewed legal sideboard plans |
| Split | Select swaps and metagame weights on development only; final uses new game seeds, held-out opponent lineages and list variants. Nulls are analysed separately from realistic effects. Freeze the six exact lists and hypotheses before any outcome is inspected |
| Initialization / budget | Fixed-policy evaluation first. Then each original/variant pair gets copies of 3 independent roots, 0/0.1M/0.5M/1M/2M adaptation checkpoints, identical training distribution/budget; confirm a selected claim with 5 new roots. Screening six pairs to 0.5M costs up to 18M games; do not adapt obvious nulls unless probing training variability |
| Opponents | Frozen diverse suite; repeat selected finalist with equally budgeted opponent adaptation and at least two reviewed sideboard policies. Report uniform weights, dated metagame weights and sensitivity over a preregistered range of plausible weights |
| Primary metric | Paired match-score difference for each of the three estimands in §5.1. Rank/sign stability across roots, budgets and opponents; interval on the difference, not two separate win-rate intervals |
| Stop / threshold | Exact serialization nulls must give identical inputs/results under canonical card-instance coupling; null aliases test identity dependence and should be reported separately. Across stochastic training nulls, require family-level equivalence within ±1pp using simultaneous intervals; unresolved precision is not a pass. All large controls must have the reviewed sign with lower confidence bound beyond 5pp. Only then interpret a realistic swap whose 95% interval excludes zero and whose effect clears a preregistered practical margin (default 1pp). Report reversals rather than forcing one ranking |
| Reproducibility | Canonical multiset and card-instance coupling, variant hashes, policy/optimizer copies, sideboard correspondence, adaptation curriculum and seed-block outcomes. Preserve all tested swaps, not only winners; use a fresh final family for confirmation |

A deliberately bad deck is not guaranteed to lose by a particular magnitude
under every poor policy. That is why failure of the large controls is diagnostic:
investigate policy proficiency and the control design before using the system
to resolve subtle card choices. For small swaps, a compute cap reached before
the powered sample size yields **inconclusive**, not “no effect.”

## 8. Cost, scheduling and the first campaign

### 8.1 Measured resources and honest extrapolation

Read-only `nvidia-smi` and `getconf _NPROCESSORS_ONLN` verified **eight NVIDIA
H100 80 GB HBM3 GPUs and 64 logical CPUs** on `h100-private`. The local review
and diagnostics ran on the Mac; its timings are not training-server benchmarks.
CPU topology, memory bandwidth, NUMA placement, workload and sharing matter;
64 logical CPUs must not be silently treated as 64 independent physical cores.

At the 14:11 snapshot, mean wall time over the last 50 ordinary 2,048-game
iterations was:

| Active R6 architecture | Seconds / iteration | Games / hour | One-GPU hours / 10M games |
|---|---:|---:|---:|
| h128 entity | 6.119 | 1.205M | 8.30 |
| h128 attention, current slow path | 37.543 | 0.196M | 50.92 |
| h256 entity | 10.725 | 0.687M | 14.55 |

These are **observed co-scheduled steady-state rates**, not reservations or
full-run averages. They exclude extra evaluations, startup/compilation,
annotation and future representation costs. Do not sum rollout and update
times when they overlap. For planning use h128 **0.6–1.2M games/hour** for
modest new inputs, plus 30% startup/evaluation contingency: about **11–22
allocated GPU-hours per 10M** on one GPU. This range is a forecast requiring a
timing pilot. At 20–24 logical CPUs per run it implies about **220–528 logical
CPU-core-hours per 10M**. Attention before optimization costs roughly 66
GPU-hours per 10M with that contingency; an optimized replacement must be
measured anew. H100/V100/TPU/Titan hardware-hours are not interchangeable.

General formula: `training_GPUh = games / measured_games_per_hour × allocated_GPUs`;
add search, encoder pretraining, evaluation and failed screens. For BO3
evaluation measure **matches/hour**, including sideboarding and recurrent
state. A planning range of 20k–100k games/hour on one allocated GPU is merely
a conservative placeholder for heterogeneous policy evaluation; replace it
after a 400-block pilot. Card-effect power can make evaluation more expensive
than training a small continuation.

### 8.2 Ranked first campaign and gates

| Order / deliverable | Budget and demand | Proceed only if |
|---|---|---|
| **F: correctness and observability dossier** | 0–6 GPUh; 80–200 logical CPU-coreh; 12–20 expert hours. About 2–4 working days including fixture adjudication, a planning estimate | Fix reproduced action omission and hidden-list contract; complete invariance and coverage gates; instrument truncation/cutoffs |
| **Timing and data manifests** | 3–12 GPUh; 60–240 CPU-coreh; 4–8 curation hours; roughly half a day of reserved machine time | Frozen legal lists, splits, roots and opponent manifest; trustworthy actual throughput; no overlap with existing jobs |
| **G core / R semantics: C vs S, three independent roots each** | 60M games: **66–132 GPUh**, roughly 1,320–3,170 CPU-coreh. Two concurrent runs: 33–66 elapsed compute hours; three only if CPU measurements justify it | List and pair-transfer signal, retention gate; do not expand to all holdout folds if neither policy shows usable transfer |
| **X initial response tests and cross-play** | 12M response games: **13–27 GPUh**, 260–650 CPU-coreh; reserve 5–20 additional GPUh for powered cross-play. 8–24 annotation hours for tactical failures | Responses learn against positive control; vulnerabilities quantified; no persistent critical tactical failure |
| **O0/O1 objective screen** | 12M continuation games: **13–27 GPUh**, 260–650 CPU-coreh; two concurrent runs about 7–14 elapsed hours | A worthwhile objective signal or a clear reason to retain current gamma |
| **Selected five-seed confirmation** | One C/S contrast at 10M each: 100M games, **110–220 GPUh**, 2,200–5,280 CPU-coreh, plus final evaluation; roughly 55–110 elapsed hours at two concurrent runs | Final held-out gain and retention; independent roots; no silent reuse of final tuning data |

The first **screening** campaign totals about **100–224 GPU-hours including
the additional powered cross-play**, approximately **2,000–4,900
logical CPU-core-hours**, and **24–52 expert/curation hours**. Allow roughly
**one to two calendar weeks on the current machine** with two reserved training
slots and annotation in parallel; existing R6 jobs or repair work can extend
that. These are capacity plans, not measurements or training commitments.
Confirmation is a separate **110–220 GPU-hour** allocation and can be rejected
by the screening gates.

The remaining programme is conditional, not an instruction to run everything:

| Follow-on | Explicit scale / indicative cost |
|---|---|
| Separate archetype/card transfer | Each two-arm, three-seed, 10M fold is another 60M games, 66–132 GPUh. All six archetype folds with five seeds and two arms would be 600M games, 660–1,320 GPUh; start with the two pilot folds |
| R representation pair | 2 arms × 3 seeds × 5M = 30M, 33–66 GPUh at h128-like rates; approximately 198 GPUh on the unoptimized attention path. Time first |
| T teaching screen | Control plus three treatments × 3 seeds × 2M = 24M PPO games, 26–53 GPUh, **plus** up to 30 GPUh BC and 60 GPUh / 1,200 CPU-coreh search teaching; 1,000 reviewed labels at 2–5 minutes each plus double review of 20%: about 40–100 annotator-hours |
| V small-swap screen | Six pairs × 2 policies × 3 roots × 0.5M = 18M games, 20–40 GPUh; 4–8 hours for lists/sideboard controls. At a 2pp target and illustrative variance above, six comparisons in one aggregate cell need ~47k–118k evaluation games; per-matchup claims multiply that workload. At 1pp, ~188k–470k games before training variance/multiplicity adjustments |
| V one selected confirmation | 2 variants × 5 independent roots × 2M = 20M games, 22–44 GPUh; opponent adaptation roughly adds a comparable training budget if both sides are adapted. Final BO3 sampling may dominate, and must be costed from measured block variance and matches/hour |
| Human strength pilot | 60 BO3 × assumed 25–45 minutes = 25–45 pilot-hours, plus roughly 8–16 hours adjudication/review; durations are estimates |

### 8.3 What two or three more machines should buy

Assuming each added machine really has eight H100s and 64 logical CPUs, the
expansion gives **three or four machines total**: 24/32 GPUs and 192/256 logical
CPUs. Verify each rather than treating this as already provisioned. At two
well-fed experiments per machine, that is six/eight concurrent learning jobs.
The first six G runs then fit in one wave of about 11–22 compute hours, versus
three waves of 33–66 hours on one machine. Shared data preparation and expert
review do not accelerate by the same factor.

Use the extra independent machines primarily for **independent seeds, ablations,
holdout folds and response learners**. That addresses today's evidence deficit.
It also avoids inter-node synchronization and gives useful results if one run
fails. Three/four machines do not justify launching every configuration or
counting correlated runs as independent.

Use the at-most-two-machine InfiniBand pair for one run only after an isolated
profile shows the learner, rather than rollout CPUs/inference, limits progress.
First measure one versus two versus four learner GPUs on one machine, including
global-batch changes and strength per game. Then measure the two-node case with
identical algorithmic settings. A 2× GPU allocation delivering 1.5× speed gives
shorter elapsed time but **33% more GPU-hours**. Accept that only when shorter
latency has a clear project value; require at least 70% parallel efficiency and
no learning-quality regression for routine use. Unlinked machines can run
independent jobs or coarse actor workloads, but do not presume synchronous
multi-node PPO efficiency over an unspecified network.

Do not buy accelerator capacity to compensate for an unmeasured CPU bottleneck.
The current slow attention path and feature previews are substantial systems
questions; #53 is a useful engineering branch to validate, not a reason to
skip the scientific controls.

## 9. What would falsify this direction?

The recommendation is deliberately conditional. The following outcomes would
change it, rather than simply justify another larger run:

1. **Simulator economics fail:** supported-deck additions repeatedly introduce
   high-impact independent-rules failures, or exact action coverage cannot be
   maintained at usable throughput. Then narrow the supported game and reassess
   integrating a mature engine before claiming generalized Pauper.
2. **Transfer fails at matched resources:** structured/deck-conditioned models
   repeatedly fail held-out lists/cards after three-seed screens and five-seed
   confirmation, or require nearly scratch-training cost for every new deck.
   Then abandon the assumption that one compact shared representation gives
   efficient generalization; test modular specialists/adapters and better
   semantic teaching. Do not call fixed-list expansion transfer.
3. **PPO's strategic ceiling persists:** corrected observations, an aligned
   objective, a diverse population and targeted instruction still fail simple
   long-horizon tasks. If search or a game-theoretic learner solves the same
   bounded tasks under a matched budget, prioritize that alternative rather
   than scaling undirected PPO again.
4. **Shared competence is unstable:** each new archetype causes >3pp retained
   losses on older tasks across independent roots despite balanced exposure.
   A shared trunk with adapters or a specialist bank then has a stronger case
   than a monolithic policy.
5. **Robustness does not improve:** stronger bots/lineage ratings repeatedly
   coexist with easily learned large responses. Then optimize population
   robustness and reconsider average-policy/equilibrium-oriented objectives.
6. **Card conclusions do not stabilize:** nulls produce effects, obvious controls
   fail, or realistic swap signs keep reversing with training maturity,
   opponent adaptation or reasonable sideboard plans. In that case use the
   model to generate hypotheses and inspect tactics; do not publish precise
   card rankings as established research results.

Passing the proposed gates would support progressively stronger claims: first
correctness within a declared supported game, then local competence, then
measured transfer and robustness, then narrowly defined card-intervention
results. It would still not establish optimal Magic play. The useful goal is
a player whose limits and uncertainty are measured well enough that its games
and experiments can be trusted for the question being asked.

## Appendix A. Reproduce the concrete diagnostics

Run from this checkout after `make setup`, using its own native extension.
The following reads engine state and constructs local scenarios; it does not
alter training runs or repository code. At the inspected commit, **both engines**
print 1,023 combat options, zero maximum defender/blocker-eleven damage, absent
legal lethal, both preview streams skipped, 64 entities, zero hand entities,
and an absent Bolt pointer. The final lines show opponent metadata entering
features, equal Brainstorm/Ponder shapes and life potential 4.

```bash
.venv/bin/python - <<'PY'
import os
from tests.helpers import scenario, choose, pass_priority
from mtg_ml.encode import entity_features, option_previews, state_features
from mtg_ml.rl.rollout import _potential

for engine in ('python', 'native'):
    os.environ['MTG_ENGINE'] = engine
    g = scenario(
        p0={'battlefield': [('Nyxborn Hydra', {'counters': 20})]},
        p1={'battlefield': ['Eldrazi Spawn'] * 11, 'life': 9},
        step='declare_attackers',
    )
    choose(g, 'Attack with Nyxborn Hydra')
    if g.decision.kind == 'declare_attacker':
        choose(g, 'Done declaring attackers')
    pass_priority(g, 2)
    for _ in range(11):
        choose(g, 'blocks Nyxborn Hydra')
    pass_priority(g, 2)
    opts = g.legal_options()
    pv = option_previews(g, 0, 6)
    print(engine, g.decision.kind, len(opts),
          max(o.key[1][-1] for o in opts),
          max(o.key[1][10] for o in opts),
          any(tuple(o.key[1]) == (1,) * 11 + (9,) for o in opts),
          all('pv:sim:skipped' in p for p in pv),
          all('pv:simp:skipped' in p for p in pv))
    cap = scenario(p0={'battlefield': ['Eldrazi Spawn'] * 64,
                       'hand': ['Lightning Bolt']},
                   p1={'battlefield': ['Island']})
    ents, index = entity_features(cap, 0, 6)
    print('cap', len(ents), sum('e:zone:hand' in e for e in ents),
          cap.players[0].hand[0].oid in index)
    a = scenario(deck_names=('jund', 'blue'))
    b = scenario(deck_names=('jund', 'elves'))
    print('metadata', set(state_features(a, 0, 6)) ^
                      set(state_features(b, 0, 6)))
    c = scenario(p0={'hand': ['Brainstorm', 'Ponder'], 'life': 100},
                 p1={'life': 20})
    x, y = c.players[0].hand
    print('shape', tuple(x.face.shape) == tuple(y.face.shape),
          'potential', _potential(c, 0))
PY
```

The separate targeted existing-test command was:

```bash
.venv/bin/python -m pytest -q \
  tests/test_training_knobs.py tests/test_sim_previews.py tests/test_rl.py \
  -k 'finish or value_clamp or per_turn or hidden or skipped or logits_masked'
```

Result on the review Mac: **22 passed, 60 deselected, 29.11 seconds**.
No Rust or Python implementation changes were made for this review. Passing
tests coexist with the reproduced limitations because the present tests permit
those documented approximations/contracts. The full suite and a new training
campaign were not run.

## Appendix B. Evidence retention and reading map

Local gitignored evidence lives in `.context/generalized-review/`: primary-paper
PDFs/text and `papers/manifest.json` (URLs and SHA-256),
`remote-snapshot.json` (timestamped run metrics), `hardware-fs4.txt`, and
`diagnostics.py` / `diagnostics.json`. These files are working evidence, not
portable published artifacts. The critical reproduction, checkpoint identities,
source links and protocol are retained in this report so it does not depend
on those local files. No large checkpoints were copied into Git.

Primary sources are linked beside the claims they support. Further useful
reading details: PPO's clipped objective and masking paper for policy math;
NFSP's supervised average-policy/reservoir construction; PSRO's empirical
payoff game and approximate responses; DeepNash's regularized dynamics and
compute appendix; ReBeL §9 and appendix E for limitations/compute; Student of
Games' GT-CFR, soundness and per-game experiments; PerfectDou's training-only
information and compute appendix; Baisero–Amato's history/state critic analysis;
Expert Iteration's Hex experiments; DAgger's learner-induced distribution;
Procgen's held-out content; and Agarwal et al.'s run-level intervals and
performance profiles. These support mechanisms and experimental standards,
not a claim that their benchmark success will transfer unchanged to Magic.

For shaping, the original theoretical reference is
[Ng, Harada and Russell, 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).
The telescoping argument needs consistent discounts and terminal handling;
the specific life-potential clipping issue in §4 is a finding about this code.
The report's feasibility judgement, numerical gates and budget forecasts are
independent recommendations, explicitly conditional on the proposed tests.


### Archived weight identities

These are the full `final.pt` digests read from each archive's `SHA256`, not
newly computed checksums of changing live files. A final-weight identity does
not turn an earlier in-training metric into a final evaluation: R4/R5 metric
checkpoints remain at the game counts stated in §1.2. Some archived code labels
identify a branch plus fixes instead of one commit; that unresolved provenance
is retained literally below. No exact SHA is inferred from the current branch.

| Archive | Recorded code label | `final.pt` SHA-256 |
|---|---|---|
| `20261006-overnight-selfplay` | claude/overnight-jund-blue-run-bf922d @ a3a70b7 (PR #12) on main 2fccf29 | `28b39e6a18b7b96dd48dd236c191654f8bc1023716457c73870ad6f8771f6887` |
| `20261007-r1-control` | claude/next-integration @ f953f1e | `0d8d8b675dd77db23e7f065d2f4beb5f8b63cf0070aa6194c109926617a35d87` |
| `20261007-r1-lranneal` | claude/next-integration @ f953f1e | `559944ba95798f9f852153767c17e09a2b5152e8a2dbbb9cf1ea6b83229765df` |
| `20261007-r1-gammaturn` | claude/next-integration @ f953f1e | `af7f04c0560b8260e71c5b27e59bcac9653ad577f89dfee90605db70b88bebb4` |
| `20261007-r1-exploit-blue` | claude/next-integration @ f953f1e | `b98060b39e1221cf7834a98af5b5d3f585a3b82832810255548720d73cf62a53` |
| `20261007-r1-exploit-jund` | claude/next-integration @ f953f1e | `b0a4b86b1f16d4a7a30720f27e8019619d2531106f2b71807f45d062c8ca12bf` |
| `20261007-r3-attn` | claude/next-integration-2 @ bdad997 + PR #22 | `b8b64dc4584b84069a3dba986ebf56b0e4c6d1e2a48517684ec85a48f35dfb55` |
| `20261007-r3-control` | claude/next-integration-2 @ bdad997 + RNG restore fix (PR #22) | `369890046c903abd99310bd0ef24892554727c7787061c067c0dfbe13a607d4d` |
| `20261007-r4-mix` | PR #28 claude/multi-matchup (+ stack fix) | `a5a2bed176b5bb193c977713e2b9ea0ff1784b8fb0ad22bb65a65ecbf8079b00` |
| `20261007-r4-control` | PR #28 claude/multi-matchup (+ stack fix) | `56ab6f1e7d70b44b33dccb0f665584c91a8e8db0eb154e7270c28254e8cfd1c1` |
| `20261007-r5-mix-h128` | PR #28 claude/multi-matchup (+ stack fix) | `cdfe14f1fe64628fbf2b3a178e2c6853178796fa6bf6d15a808532b0a31ad85b` |
| `20261007-r5-mix-h256` | PR #28 claude/multi-matchup (+ stack fix) | `8a4beda9e5ad8432b6be2067464fc20826c7bc56347b6be3f235d4557b438a0d` |
| `20261007-r5-mix-h256-lr3e-4` | PR #28 claude/multi-matchup @ 2026-10-07 | `d5c7b497941b76342c5ffa4db50e6960124f44168f82ebbdadb6c83c961b294c` |
| `20261007-red-madness-1m` | claude/mono-red-burn-pauper-627e45 @ 09020f2 (PR #23; flags --opponent/--init-from/--learner-seat before the merge renamed them to --exploit/--exploit-deck) | `f6023fee7e5803133b6f6a46accd491b909f6cbbc279715039015a14433b578e` |
| `20261007-fs4-ft` | claude/feature-set-4 @ 759358e (PR #33) | `a560f282824e6a5ec357b9b677a69840029d2571571896785ea5d3d5d220a198` |
