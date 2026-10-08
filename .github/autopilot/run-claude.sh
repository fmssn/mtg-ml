#!/usr/bin/env bash
# Keep the four/five positional arguments used by older PR workflow files.
set -euo pipefail
exec python "$(dirname "$0")/runner.py" "$@"
