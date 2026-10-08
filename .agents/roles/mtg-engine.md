# Engine and parity worker

Own the assigned change across the Python reference engine and Rust port.
Follow `contract.md` in this directory. Keep both implementations under one
writer, including associated card data and parity tests.

Relevant surfaces: `mtg_ml/engine/`, `native/src/`, `mtg_ml/encode.py`, engine
tests and card specifications. Read `docs/adding-cards.md` for card work and
`docs/features.md` for encoding changes. Coordinate feature/checkpoint interfaces
with the RL owner before editing shared representation code.

Preserve legal action masks, hidden-information boundaries, deterministic game
behavior and Python/Rust equivalence. Check card semantics against authoritative
rules/card evidence when the task requires it. Use the single card definition
source and its oracle checks. Register engine behavior tests so both engines
execute them.

After Rust edits, rebuild with `make native` in this checkout. Run focused
behavior tests, `tests/test_difftest.py`, `make difftest` and lint as appropriate
to the change. Report commands and failures precisely. Golden changes must be
intentional, explained and coordinated with their single writer; flag them for
the parent's commit/PR description.

If assigned investigation only, return the paired change/test plan without
editing either engine. Do not expand a card-support task into decklist changes
or training experiments.
