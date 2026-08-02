#!/usr/bin/env bash
# End-to-end toy run of the Bench source: convert a few scene groups (every
# run writes all 13 qa_category partitions), then render both pipeline ends.
#
# Usage:
#   scripts/run_bench.sh [max_records]
#
#   max_records   scene groups to convert (default 2; -1 = all 409). Each
#                 group yields 13 records, one per partition, so even 2
#                 groups cover every partition and keep the composite
#                 metrics (which join a group across horizons) computable.
set -euo pipefail

MAX_RECORDS=${1:-2}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
SCRIPTS=$HERE/scripts

cd -- "$HERE"

echo "==> converting bench ($MAX_RECORDS scene groups)"
"$PYTHON" convert_strideqa.py \
  --source bench --max_records "$MAX_RECORDS" \
  --mirror_dir mirror --output_dir out

PYTHON=$PYTHON "$SCRIPTS/visualize_bench.sh" \
  input mirror out/bench_input.html 5
PYTHON=$PYTHON "$SCRIPTS/visualize_bench.sh" \
  output out out/bench_output.html 5
