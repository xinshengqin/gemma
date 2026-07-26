#!/usr/bin/env bash
# Render k examples from a pipeline OUTPUT Bagz file to a standalone HTML page.
#
# Usage:
#   scripts/visualize_ppl_output.sh <bagz-path-or-spec> <output.html> [k]
#
# The first argument is either a single shard (`golden.bagz`) or a sharded spec
# (`msop765k_test@8.bagz`). Fewer than k records in the file renders all of them.
#
# Example:
#   scripts/visualize_ppl_output.sh out/msop765k_test@1.bagz /tmp/out.html 5
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,12p' "$0" >&2
  exit 64
fi

BAGZ=$1
HTML=$2
K=${3:-5}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}

if [[ ! -e $BAGZ ]]; then
  # A sharded spec such as name@8.bagz has no file of its own; check shard 0.
  probe=${BAGZ/@*.bagz/-00000-of-*.bagz}
  compgen -G "$probe" >/dev/null || {
    echo "no such Bagz file or spec: $BAGZ" >&2
    exit 66
  }
fi

cd -- "$HERE"
exec "$PYTHON" inspect_msop765k.py \
  --stage output --bagz "$BAGZ" --k "$K" --html "$HTML"
