# One entry point for the common tasks. Works in any worktree; CARGO_TARGET_DIR is
# shared so each worktree does not rebuild the Rust engine from scratch.
PY ?= python
export CARGO_TARGET_DIR ?= $(abspath $(shell git rev-parse --git-common-dir)/..)/.cargo-target

.PHONY: setup native test test-fast lint lint-fix difftest golden

setup:            ## editable install with dev + rl extras, then the native engine
	uv pip install -e '.[dev,rl]' ruff "maturin>=1.5,<2"
	$(MAKE) native

native:           ## build the Rust engine into the active environment
	cd native && maturin develop --release

test:             ## full suite (both engines if native is built)
	$(PY) -m pytest -q

test-fast:        ## skip tests marked slow
	$(PY) -m pytest -q -m "not slow" -x

lint:
	ruff check .

lint-fix:         ## apply ruff's safe autofixes (no formatter yet: the tree is not ruff-formatted)
	ruff check --fix .

difftest:         ## Python vs Rust engine in lockstep
	$(PY) -m mtg_ml.difftest fuzz --games 2000 --jobs 8

golden:           ## re-record golden digests -- only when game behaviour changes on purpose
	$(PY) -m mtg_ml.trace record --games 210
