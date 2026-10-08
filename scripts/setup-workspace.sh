#!/usr/bin/env bash
# Bootstrap only this checkout's environment; never install into global Python.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"

for tool in uv cargo; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Missing %s. Install uv and the Rust toolchain, then run make setup.\n' "$tool" >&2
        exit 1
    fi
done

if [[ ! -x .venv/bin/python ]]; then
    uv venv --python "${MTG_PYTHON:-3.11}" .venv
fi
source .venv/bin/activate
# Cloud development uses CPU torch, matching CI. Local Macs get the normal wheel.
if [[ "${CONDUCTOR_IS_LOCAL:-1}" == 0 ]]; then
    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
fi
uv pip install --python .venv/bin/python -e '.[dev,rl]' ruff 'maturin>=1.5,<2'
make native
printf 'Workspace ready. Use make test-fast, make lint, or .venv/bin/python.\n'
