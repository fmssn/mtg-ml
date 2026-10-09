# One entry point for the common tasks. Works in any worktree; CARGO_TARGET_DIR is
# shared so each worktree does not rebuild the Rust engine from scratch.
PY ?= $(if $(wildcard .venv/bin/python),$(abspath .venv/bin/python),python)
RUFF ?= $(if $(wildcard .venv/bin/ruff),$(abspath .venv/bin/ruff),ruff)
export PATH := $(HOME)/.cargo/bin:$(PATH)
export CARGO_TARGET_DIR ?= $(abspath $(shell git rev-parse --git-common-dir)/..)/.cargo-target

.PHONY: setup native test test-fast lint lint-fix difftest golden replay live \
	test-remote test-fast-remote difftest-remote

setup:            ## create an isolated workspace environment and build native
	bash scripts/setup-workspace.sh

native:           ## build the Rust engine into the active environment
	@if [ -x .venv/bin/python ]; then . .venv/bin/activate; fi; \
	cd native && maturin develop --release --locked

test:             ## full suite (both engines if native is built)
	$(PY) -m pytest -q

test-fast:        ## skip tests marked slow
	$(PY) -m pytest -q -m "not slow" -x

lint:
	$(RUFF) check .

lint-fix:         ## apply ruff's safe autofixes (no formatter yet: the tree is not ruff-formatted)
	$(RUFF) check --fix .

difftest:         ## Python vs Rust engine in lockstep
	$(PY) -m mtg_ml.difftest fuzz --games 2000 --jobs 8

test-remote:      ## full suite on h100-private (ARGS="tests/test_rl.py -k foo")
	bash scripts/remote-test.sh test $(ARGS)

test-fast-remote: ## test-fast on h100-private
	bash scripts/remote-test.sh test-fast $(ARGS)

difftest-remote:  ## differential fuzz on h100-private
	bash scripts/remote-test.sh difftest $(ARGS)

golden:           ## re-record golden digests -- only when game behaviour changes on purpose
	$(PY) -m mtg_ml.trace record --games 210

replay:           ## replay viewer at CONDUCTOR_PORT (8765 outside Conductor)
	bash scripts/run-workspace.sh replay

live:             ## play vs MTG_MODELS_DIR at CONDUCTOR_PORT + 1
	bash scripts/run-workspace.sh live
