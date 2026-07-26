#!/usr/bin/env bash
# Render k examples from the pipeline INPUT -- the raw mirror -- to a standalone
# HTML page. Fields are the untransformed parquet values, and each image is read
# from its per-label tarball, so this page can be compared against the output
# page to see exactly what the conversion changed.
#
# Usage:
#   scripts/visualize_ppl_input.sh <mirror-dir> <output.html> [k] [split]
#
# The mirror must already hold <split>.parquet and the shards for the rows that
# get sampled; nothing is downloaded.
#
# Example:
#   scripts/visualize_ppl_input.sh mirror /tmp/in.html 5 test
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,15p' "$0" >&2
  exit 64
fi

MIRROR=$1
HTML=$2
K=${3:-5}
SPLIT=${4:-test}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}

# Resolve before cd, so a relative mirror path is read from the caller's cwd.
MIRROR=$(cd -- "$MIRROR" && pwd) || {
  echo "no such mirror directory: $1" >&2
  exit 66
}
if [[ ! -f $MIRROR/$SPLIT.parquet ]]; then
  echo "mirror has no $SPLIT.parquet: $MIRROR" >&2
  exit 66
fi

cd -- "$HERE"
exec "$PYTHON" inspect_msop765k.py \
  --stage input --mirror_dir "$MIRROR" --split "$SPLIT" --k "$K" --html "$HTML"
