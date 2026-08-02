#!/usr/bin/env bash
# Render k Bench examples from one end of the pipeline to a standalone HTML
# page: the 4-frame filmstrip with the Set-of-Marks region overlaid beside
# the question and ground truth.
#
# Usage:
#   scripts/visualize_bench.sh <input|output> <dir> <output.html> [k]
#
#   input|output  which pipeline end: the raw mirror, or the converted Bagz
#   dir           mirror directory (input) or output directory (output)
#   k             examples to sample (default 5). The output end samples
#                 across all 13 partition specs.
#
# Example:
#   scripts/visualize_bench.sh output out /tmp/bench.html 5
set -euo pipefail

if [[ $# -lt 3 ]]; then
  sed -n '2,15p' "$0" >&2
  exit 64
fi

STAGE=$1
DIR=$2
HTML=$3
K=${4:-5}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}

DIR=$(cd -- "$DIR" && pwd) || {
  echo "no such directory: $2" >&2
  exit 66
}

cd -- "$HERE"
case $STAGE in
  input)
    exec "$PYTHON" inspect_strideqa.py \
      --stage input --source bench --mirror_dir "$DIR" --k "$K" --html "$HTML"
    ;;
  output)
    exec "$PYTHON" inspect_strideqa.py \
      --stage output --source bench --output_dir "$DIR" --k "$K" --html "$HTML"
    ;;
  *)
    echo "stage must be 'input' or 'output', got: $STAGE" >&2
    exit 64
    ;;
esac
