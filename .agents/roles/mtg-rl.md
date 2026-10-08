# RL and training worker

Handle the assigned features, model, rollout, inference, PPO or checkpoint
change. Follow `contract.md` in this directory.

Start from `mtg_ml/rl/` and the supplied tests. Read `docs/features.md` before
changing feature versions or checkpoint compatibility. Encoding shared with
Rust belongs to a coordinated engine task; agree on the interface and one
writer before touching it. Preserve legal masking, seat-visible observations,
recurrent state, rollout alignment and resume semantics.

Use focused CPU checks first. GPU-only tests must be marked appropriately and
require a current explicit device/CPU allocation. Report numerical tolerances,
precision and hardware for performance or parity claims. A microbenchmark does
not establish end-to-end throughput or playing strength.

For an authorized experiment, consult `docs/context.md`,
`docs/experiments/README.md` and the ladder protocol. Record exact code ref,
parent, configuration, sampled/greedy results, ladder score and verdict. Agree
with the parent who writes the ledger. Keep checkpoints and run artifacts out
of Git, preserve existing jobs and archives, and arm unattended jobs as scripts
independent of the agent session.

Separate implementation completion from outstanding measurement. Return the
remaining resource/validation requirement rather than claiming an unmeasured
optimization target was met. Do not launch a campaign solely because the code
is ready.
