# Reinforcement Learning for Magic: The Gathering (Pauper) — Research Summary

> Status: research notes, October 2026. Internal research use.
> Terminology: per org convention, "AI" is referred to as **SI** ("Übelste Intelligenz", **ÜI** in German).

## 1. Goal

Train an agent for Magic: The Gathering via **reinforcement learning through self-play**, starting with **two fixed Pauper decks** playing against each other. The focus is explicitly statistical/learning-based (RL), not deterministic solving or exhaustive calculation of lines.

## 2. Feasibility verdict

Feasible as a scoped research project.

- The hard part is **not** the learning algorithm but a **rules engine** that is correct, fast, and scriptable from a training loop.
- Exact solving is impossible in principle: optimal play in MTG is at least as hard as the Halting Problem (Churchill, Biderman & Herrick). This is an argument *for* learned, approximate policies, i.e. RL.
- Two fixed decklists strongly reduce difficulty: fixed card vocabulary, known opponent decklist, hidden information limited to hand contents and library order.
- Prior evidence that RL works on MTG:
  - MTG-Causal-RL (2026): masked PPO reached ~68% win rate (Mono-Red Aggro) against its opponent pool; clearly above random.
  - MageZero (hobby project on XMage): a tempo deck went from 16% win rate (built-in minimax AI) to 66% after RL.
  - DraftZero experiment (built on MageZero): 2,507 self-play games, one rented L40S GPU, ~34 h, ~$28.
- Compute: a single GPU suffices for the model. The bottleneck is **simulator throughput** (games per second), so many CPU cores for parallel environments matter more than GPU size.

## 3. Rules engines

| Engine | License | Card coverage | RL interface | Notes |
|---|---|---|---|---|
| **XMage** ([magefree/mage](https://github.com/magefree/mage)) | MIT | 32,000+ unique cards, Pauper mode supported | Via MageZero fork | Most tested open-source engine (~9,000 unit tests, ~80% coverage). Java. Recommended base. |
| **MageZero** ([WillWroble/MageZero](https://github.com/WillWroble/MageZero)) | see repo | Uses XMage | AlphaZero-style framework (MCTS + NN, PUCT), one agent per deck | Alpha quality, essentially one maintainer. |
| **Forge** ([Card-Forge/forge](https://github.com/Card-Forge/forge)) | GPL-3.0 | Very broad | Headless sim possible, but heavy (one JVM per game) | Best used as **independent benchmark opponent** / referee, not training engine. |
| **Argentum Engine** ([wingedsheep/argentum-engine](https://github.com/wingedsheep/argentum-engine)) | not verified, check LICENSE | Only Portal, Onslaught block, Khans, Dominaria | Excellent: Gym-style API, immutable state with O(1) fork, batch stepping, hidden info masked, HTTP server for Python, JVM-side AlphaZero trainer | Pauper cards would have to be written in its Kotlin card DSL. |
| **pauper_sim** ([JensJansen/pauper_sim](https://github.com/JensJansen/pauper_sim)) | see repo | Curated subset; implements madness, flashback, stack, priority, mulligans | Attention-based self-play RL, per-deck policies, league vs. historical opponents | Closest to this project's goal, **but** a one-person, AI-assisted hobby project. Reference for ideas only, not a trusted foundation. |
| **mtg_ai_engine** ([sheigl/mtg_ai_engine](https://github.com/sheigl/mtg_ai_engine)) | see repo | Claims full CR enforcement | Python REST API, logs state / legal actions / chosen action per priority | Unvalidated. |
| **MTG-Causal-RL env** ([anonymous repo](https://anonymous.4open.science/r/mtg-causal-rl-E927)) | see repo | 56 unique Standard-2025 cards | Gymnasium, 3,077-dim observation, 478 masked actions | Anonymized review link: clone early, it may move. |

### Build our own engine?

- **Upside:** an engine that only knows the ~20–30 unique cards of two decks could be 100–1000× faster than XMage, with full control over state/action encoding.
- **Risk:** rules bugs. RL agents reliably find and exploit them, producing a policy that is "strong" in a broken game. Madness, triggers and the stack are typical bug sources.
- **Recommended path:**
  1. Prove learning on XMage / MageZero first.
  2. If throughput is the bottleneck, write a minimal engine for the two decks.
  3. **Differential testing:** replay identical seeds and action sequences in both engines; switch only when they agree.
  4. Keep Forge as the final independent opponent.

## 4. Academic literature

The literature is thinner than for Go, poker or Hearthstone. Most work covers drafting, deckbuilding or classic tree search. **No peer-reviewed paper was found showing an AlphaZero-style or pure-RL self-play agent reaching strong play in full-rules MTG** — a genuine gap.

### Gameplay
- Ward & Cowling (2009), *Monte Carlo search applied to card selection in Magic: The Gathering*, IEEE CIG.
- Cowling, Ward & Powley (2012), *Ensemble Determinization in Monte Carlo Tree Search for the Imperfect Information Card Game Magic: The Gathering*, IEEE TCIAIG. [eprints](https://eprints.whiterose.ac.uk/75050/) — foundational; fixed decks, determinization for hidden info, binary decomposition of move generation. **Read first.**
- Da Costa Cunha, Mian, French & Liu (2026), *Causal Reinforcement Learning for Complex Card Games: A Magic The Gathering Benchmark*. [arXiv:2605.06066](https://arxiv.org/abs/2605.06066)
- *Dynamic Resource Allocation for Ensemble Determinization MCTS* (2026). [arXiv:2607.13007](https://arxiv.org/abs/2607.13007) — builds on Cowling et al.; MTG-specificity not verified.

### Theory
- Churchill, Biderman & Herrick, *Magic: The Gathering is Turing Complete* (FUN 2021). [arXiv:1904.09828](https://arxiv.org/abs/1904.09828) — optimal play undecidable; legality of blocks is coNP-complete.

### Adjacent (not gameplay)
- *AI solutions for drafting in Magic: The Gathering* (IEEE CoG 2021). [arXiv:2009.00655](https://arxiv.org/abs/2009.00655)
- *Learning With Generalised Card Representations for "Magic: The Gathering"* (2024). [arXiv:2407.05879](https://arxiv.org/abs/2407.05879)
- Deckbuilding with a genetic algorithm, Master's thesis, NTNU (2017).
- Vieira, Chaimowicz & Tavares, *Reinforcement Learning in Collectible Card Games: Preliminary Results on Legends of Code and Magic* (SBGames 2019). [PDF](https://www.sbgames.org/sbgames2019/files/papers/ComputacaoShort/198299.pdf)

### Beyond MTG: methods for the roadmap
Grouped by the open problems in this project (search targets, expensive `fork()`, masked PPO, bigger trunk). ★ = read first.

**Search under hidden information** (for `view.determinize()` + search targets)
- Long, Sturtevant, Buro & Furtak (2010), *Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search*, AAAI. Explains when determinization breaks (strategy fusion, non-locality).
- Cowling, Powley & Whitehouse (2012), *Information Set Monte Carlo Tree Search*, IEEE TCIAIG. The standard successor to ensemble determinization.
- ★ Brown, Bakhtin, Lerer & Gong (2020), *Combining Deep Reinforcement Learning and Search for Imperfect-Information Games* (ReBeL). [arXiv:2007.13544](https://arxiv.org/abs/2007.13544)
- Schmid et al. (2023), *Student of Games*, Science. [arXiv:2112.03178](https://arxiv.org/abs/2112.03178)

**Sample-efficient search distillation** (cheap search when `fork()` is expensive)
- Anthony, Tian & Barber (2017), *Thinking Fast and Slow with Deep Learning and Tree Search* (Expert Iteration). [arXiv:1705.08439](https://arxiv.org/abs/1705.08439)
- ★ Danihelka, Guez, Schrittwieser & Silver (2022), *Policy Improvement by Planning with Gumbel* (Gumbel MuZero), ICLR. [OpenReview](https://openreview.net/forum?id=bERaNdoegnO) — guaranteed policy improvement with very few simulations.
- Hubert et al. (2021), *Learning and Planning in Complex Action Spaces* (Sampled MuZero). [arXiv:2104.06303](https://arxiv.org/abs/2104.06303)

**Card games with large action spaces**
- Zha et al. (2021), *DouZero: Mastering DouDizhu with Self-Play Deep RL*, ICML. [arXiv:2106.06135](https://arxiv.org/abs/2106.06135) — action encoding comparable to `Option.key`.
- ★ Guan et al. (2022), *PerfectDou: Dominating DouDizhu with Perfect Information Distillation*. [arXiv:2203.16406](https://arxiv.org/abs/2203.16406) — critic sees hidden cards during training only; cheap win for the value head.
- Li et al. (2020), *Suphx: Mastering Mahjong with Deep RL* (oracle guiding). [arXiv:2003.13590](https://arxiv.org/abs/2003.13590)
- Xiao, Zhang, Huang, Huang, Chen & Sun (2023), *Mastering Strategy Card Game (Hearthstone) with Improved Techniques*, IEEE CoG. [arXiv:2303.05197](https://arxiv.org/abs/2303.05197) — closest published analogue to this project: end-to-end policy plus optimistic smooth fictitious play on a full commercial CCG.
- *Learning to Beat ByteRL: Exploitability of Collectible Card Game Agents* (2024). [arXiv:2404.16689](https://arxiv.org/abs/2404.16689) — exploitability as an evaluation lens.

**PPO practice**
- Huang & Ontañón (2020), *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*. [arXiv:2006.14171](https://arxiv.org/abs/2006.14171)
- Huang et al. (2022), *The 37 Implementation Details of Proximal Policy Optimization*, ICLR Blog Track. [blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- Andrychowicz et al. (2021), *What Matters in On-Policy RL? A Large-Scale Empirical Study*. [arXiv:2006.05990](https://arxiv.org/abs/2006.05990)

**Self-play populations, opponent modelling, architecture**
- Vinyals et al. (2019), *Grandmaster level in StarCraft II using multi-agent RL* (AlphaStar), Nature — league training; transformer entity encoder.
- Berner et al. (2019), *Dota 2 with Large Scale Deep RL* (OpenAI Five). [arXiv:1912.06680](https://arxiv.org/abs/1912.06680)
- Heinrich & Silver (2016), *Deep RL from Self-Play in Imperfect-Information Games* (NFSP). [arXiv:1603.01121](https://arxiv.org/abs/1603.01121)
- Lanctot et al. (2017), *A Unified Game-Theoretic Approach to Multiagent RL* (PSRO). [arXiv:1711.00832](https://arxiv.org/abs/1711.00832)
- He et al. (2016), *Opponent Modeling in Deep RL* (DRON). [arXiv:1609.05559](https://arxiv.org/abs/1609.05559) — background for the opponent-action head.

**Talks** (no substantive MTG-specific RL videos found)
- ★ Noam Brown, *ReBeL* (Simons Institute). [YouTube](https://www.youtube.com/watch?v=-b33wavGOOw)
- David Silver, UCL RL course — [L7 Policy Gradient](https://www.youtube.com/watch?v=KHZVXao4qXs), [L8 Integrating Learning and Planning](https://www.youtube.com/watch?v=ItMutbeOHtc), [L10 Classic Games](https://www.youtube.com/watch?v=kZ_AUmFcZtk) ([playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)).
- *PPO Implementation: 11 Core Implementation Details*. [YouTube](https://www.youtube.com/watch?v=MEt6rrxH8W4)
- *MuZero* paper walkthrough. [YouTube](https://www.youtube.com/watch?v=We20YSAJZSE)
- AlphaStar league training [explainer](https://www.youtube.com/watch?v=BTLCdge7uSQ); Oriol Vinyals, *From AlphaGo to AlphaStar and beyond* [Part I](https://www.youtube.com/watch?v=IjZLZSZxvIs).

### Assessment of the two most promising sources
- **MTG-Causal-RL:** use as a **blueprint**, not as the engine. Copy:
  - action design (478 actions across 16 categories, typically ~15 legal at a time, masked),
  - evaluation protocol (paired seeds, bootstrap CIs, Holm-Bonferroni correction),
  - baseline: plain PPO ~68% vs. causal variant ~71%, difference not significant → causal component optional.
- **Legends of Code and Magic:** simplified card game, **not MTG**. Useful only as a fast sandbox to debug the training pipeline; results and mechanics will not transfer.

## 5. Deck selection (Pauper)

Meta snapshot (mtgdecks.net, August 2026): Mono Red Madness (~9%), Grixis Affinity (~8%), Mono Blue Terror, Jund Wildfire, Mono Red Rally, UB Faeries, Elves, Golgari Gardens.

- **Start with linear decks with few decisions per turn**, e.g. a red aggro deck vs. White Weenie or Bogles.
- **Avoid initially:** Affinity, Spy Combo, Familiars, Elves, Gates — loops, mana tricks and very large action spaces.
- **Second step:** Mono Red Madness (most-played deck; discard/madness timing is an interesting skill to learn).

## 6. RL methods overview

### Classification axes
- Model-free vs. model-based
- Value-based vs. policy-based vs. actor-critic
- On-policy vs. off-policy
- Single-agent vs. multi-agent
- Perfect vs. imperfect information (MTG: imperfect)

### Algorithm families

| Family | Examples | Fit for this project |
|---|---|---|
| Value-based | Q-learning, DQN, Rainbow, R2D2 | Weak: deterministic argmax policies are exploitable; MTG rewards mixed strategies. |
| Policy gradient / actor-critic | REINFORCE, A2C/A3C, **PPO**, **IMPALA** (V-trace) | **Start here.** Native action masking, mixed strategies, scales across CPU workers. IMPALA-style distributed setup suits slow simulators. |
| Continuous control | SAC, TD3, DDPG | Not relevant. |
| Model-based / search | AlphaZero, MuZero, Dreamer | Strong but heavy (search multiplies simulator calls), hidden info complicates planning. Phase-2/3 comparison. |
| Game-theoretic / multi-agent | Self-play, fictitious self-play, **opponent pools**, league training (AlphaStar), PSRO, NFSP, CFR/Deep CFR, ReBeL, **R-NaD** (DeepNash) | Central to this project. R-NaD reached expert Stratego with pure model-free RL and no search — closest precedent. CFR scales poorly to MTG. |
| Offline / imitation | Behavior cloning, CQL, IQL, Decision Transformer | Optional warm start from Forge-vs-Forge games; inherits Forge AI habits. |
| Hierarchical | Options framework | MTG decisions are nested (cast → target → mode); usually handled via architecture. |

### Network architectures
- **MLP on flat vector** — simplest; works for a fixed card pool (as in MTG-Causal-RL).
- **Recurrent (LSTM/GRU)** — memory of revealed/cast cards; important under hidden information.
- **Transformer over card tokens** — each card in each zone is a token; handles variable board sizes; current best practice.
- **Pointer networks / autoregressive heads** — choose action type, then point at a target card (AlphaStar-style).

## 7. Opponent pool (self-play stabilization)

An opponent pool is a collection of **frozen past checkpoints** of the agent. The learning agent trains against opponents sampled from the pool instead of only its latest self.

- **Why:** naive self-play in imperfect-information games tends to **cycle** (strategy A beats B, C beats A, the agent forgets how to handle B…). Training against past versions forces robustness against all strategies seen so far.
- **Mechanics:** every N games (e.g. 50,000), freeze the current policy and add it to the pool. Pool opponents never update.
- **Sampling variants:**
  - uniform over all checkpoints,
  - recency-weighted (e.g. 50% latest, 50% random past),
  - prioritized by the current agent's loss rate (AlphaStar),
  - pruning of checkpoints that are no longer informative (e.g. beaten 99% of the time).
- **Two-deck setup:** one pool per deck (Red checkpoints train the White agent and vice versa).
- **Doubles as evaluation:** win rate against fixed old checkpoints shows real progress vs. cycling.

## 8. RL design considerations for MTG

- **Hidden information → memory:** the policy needs game history (recurrent or transformer over the log), not just the current board.
- **Sparse reward:** win/loss alone is slow. Add light shaping (life difference, card advantage) and **anneal it out** so the agent optimizes for winning.
- **High variance:** draw luck dominates single games. Evaluate over many games; track Elo vs. fixed checkpoints and Forge AI with confidence intervals.
- **Policies:** one network per deck (simpler, used by most projects) vs. one deck-conditioned network.
- **Rules correctness:** any engine bug will be exploited; validate against XMage/Forge.

## 9. Recommended architecture and roadmap

**Baseline stack:** XMage (rules) → Gymnasium-style wrapper with masked actions → masked PPO, transformer over card tokens (+ history) → opponent pool per deck → distributed IMPALA-style workers → evaluation with paired seeds and bootstrap CIs.

1. Run both decklists engine-vs-engine headlessly; measure games/second.
2. Wrap the engine as a Gymnasium environment with masked actions (borrow MTG-Causal-RL's action layout).
3. Optionally debug the training loop on LOCM or pauper_sim.
4. Train masked PPO with self-play + opponent pool; sparse win/loss reward plus annealed shaping.
5. Evaluate vs. random, built-in engine AI, Forge AI, and fixed past checkpoints.
6. Phase 2: compare against R-NaD; optionally behavior-cloning warm start.
7. Phase 3: search-based comparison (AlphaZero / MageZero); custom fast engine with differential testing if throughput limits progress.
8. Later: mulligan decisions, sideboarding, more decks.

## 10. Open questions / next steps

- Final choice of the two decklists; verify every card is implemented in XMage.
- Measure XMage throughput (games/s per core) to decide whether a custom engine is needed.
- Verify licenses of Argentum Engine, MageZero, pauper_sim, MTG-Causal-RL before reusing code.
- Clone the MTG-Causal-RL anonymous repository before it moves.
- Define evaluation protocol and compute budget up front.
