#!/usr/bin/env bash
# Run tests or the differential fuzz on the shared training box instead of this Mac.
# Syncs the working tree (uncommitted changes included, gitignored files left out)
# to ~/mtg-ml-tests/<workspace>/ there, keeps a per-workspace venv and native build,
# and runs on a pinned, niced CPU set so training keeps priority.
#
#   scripts/remote-test.sh test [pytest args]       # full suite
#   scripts/remote-test.sh test-fast [pytest args]  # skip slow tests
#   scripts/remote-test.sh difftest [fuzz args]     # Python vs Rust in lockstep
#
# Env: REMOTE_HOST (h100-private), REMOTE_CPUS (48-63), REMOTE_JOBS (16),
#      REMOTE_NAME (this checkout's directory name).
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:?usage: remote-test.sh test|test-fast|difftest [args]}"
shift
case "$mode" in test | test-fast | difftest) ;; *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

host="${REMOTE_HOST:-h100-private}"
cpus="${REMOTE_CPUS:-48-63}"
jobs="${REMOTE_JOBS:-16}"
name="${REMOTE_NAME:-$(basename "$PWD")}"
dir="mtg-ml-tests/$name"

# Some tests read git history, so the remote copy is a repository at this HEAD:
# push the commit over ssh, point HEAD and the index at it, then overlay the
# working tree so uncommitted changes show up as they do here.
ssh "$host" "mkdir -p ~/$dir && git -C ~/$dir init -q && echo /.remote-stamps/ > ~/$dir/.git/info/exclude"
head=$(git rev-parse HEAD)
git push -q -f "$host:$dir" "HEAD:refs/remote-test/head"
ssh "$host" "git -C ~/$dir update-ref --no-deref HEAD $head && git -C ~/$dir read-tree $head"
# .gitignore filters keep .venv, runs/ and build output out. They do not protect
# the remote's own copies from --delete, so the venv and stamps are protected
# explicitly.
rsync -az --delete --filter='P /.venv/' --filter='P /.remote-stamps/' --exclude=/.git \
    --exclude=/.venv/ --filter=':- .gitignore' ./ "$host:$dir/"

args=""
for a in "$@"; do args+=" $(printf '%q' "$a")"; done

# shellcheck disable=SC2087  # expand the local settings into the remote script
ssh "$host" bash -s <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"
export CARGO_TARGET_DIR="\$HOME/mtg-ml-tests/.cargo-target"
export CARGO_BUILD_JOBS=$jobs OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
cd ~/$dir
mkdir -p .remote-stamps
run() { nice -n 10 taskset -c $cpus "\$@"; }

# The autopilot workflow tests shell out to jq, which the box lacks.
if ! command -v jq >/dev/null; then
    curl -fsSL -o ~/.local/bin/jq https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-linux-amd64
    chmod +x ~/.local/bin/jq
fi
if [[ ! -x .venv/bin/python ]]; then uv venv -q --python 3.11 .venv; fi
deps=\$(sha256sum < pyproject.toml)
if [[ ! -x .venv/bin/pytest || "\$deps" != "\$(cat .remote-stamps/deps 2>/dev/null)" ]]; then
    echo "remote: installing dependencies"
    uv pip install -q --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install -q --python .venv/bin/python -e '.[dev,rl]' 'maturin>=1.5,<2'
    echo "\$deps" > .remote-stamps/deps
    rm -f .remote-stamps/native
fi
native=\$(find native mtg_ml/engine/cards.toml -type f -not -path '*/target/*' | sort | xargs sha256sum | sha256sum)
if [[ "\$native" != "\$(cat .remote-stamps/native 2>/dev/null)" ]]; then
    echo "remote: building native engine"
    (source .venv/bin/activate && cd native && run maturin develop -q --release --locked)
    echo "\$native" > .remote-stamps/native
fi

echo "remote: $mode on \$(hostname) cpus $cpus, $jobs jobs"
case "$mode" in
    test) run .venv/bin/python -m pytest -q -n $jobs -m "not gpu"$args ;;
    test-fast) run .venv/bin/python -m pytest -q -n $jobs -m "not slow and not gpu"$args ;;
    difftest) run .venv/bin/python -m mtg_ml.difftest fuzz --games 2000 --jobs $jobs$args ;;
esac
EOF
