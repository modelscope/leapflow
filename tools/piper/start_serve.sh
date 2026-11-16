#!/usr/bin/env bash
# Compatibility path; dry-run by default, with exact preview-bound service approval.
set -euo pipefail
root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
python="${PIPER_NODE_PYTHON:-$HOME/miniconda3/envs/leap/bin/python}"
exec "$python" "$root/piper.py" node start "$@"
