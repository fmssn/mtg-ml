#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-replay}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ ! -x .venv/bin/python ]]; then
    printf 'Run make setup before starting the viewer.\n' >&2
    exit 1
fi
port="${CONDUCTOR_PORT:-8765}"
case "$mode" in
    replay)
        exec .venv/bin/python -m mtg_ml.replay serve --host 127.0.0.1 --port "$port" "$@"
        ;;
    live)
        models="${MTG_MODELS_DIR:-runs}"
        if [[ ! -d "$models" ]]; then
            printf 'No model directory: %s. Set MTG_MODELS_DIR to a directory containing checkpoints.\n' "$models" >&2
            exit 1
        fi
        exec .venv/bin/python -m mtg_ml.replay serve --host 127.0.0.1 --port "$((port + 1))" \
            --models "$models" --engine "${MTG_ENGINE:-native}" "$@"
        ;;
    *)
        printf 'Usage: bash scripts/run-workspace.sh replay|live [serve options]\n' >&2
        exit 1
        ;;
esac
