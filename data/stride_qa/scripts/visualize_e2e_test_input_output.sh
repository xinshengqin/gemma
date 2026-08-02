#!/usr/bin/env bash
# Render both ends of the end-to-end test fixture, so what the golden tests
# compare can be eyeballed rather than inferred from binaries.
#
# The output end is produced by running the real converter over the fixture
# mirror into a temporary directory — the exact run the golden tests perform,
# and offline for the same reason: the fixture mirror is pre-populated. When
# the tests pass, what this renders IS the golden, with the metadata sidecars
# the goldens themselves do not keep.
#
# Writes, under testdata/visualization/:
#   e2e_dataset_spatial_{input,output}.html
#   e2e_dataset_spatiotemporal_{input,output}.html
#   e2e_bench_{input,output}.html
#
# Usage:
#   scripts/visualize_e2e_test_input_output.sh [k]
set -euo pipefail

K=${1:-5}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPTS=$HERE/scripts
MIRROR=$HERE/testdata/mirror
OUT=$HERE/testdata/visualization
PYTHON=${PYTHON:-python3}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mkdir -p -- "$OUT"
cd -- "$HERE"

for CATEGORY in ego_centric_spatial_qa ego_centric_spatiotemporal_qa; do
  SHORT=${CATEGORY#ego_centric_}
  SHORT=${SHORT%_qa}
  "$PYTHON" convert_strideqa.py \
    --source dataset --split val --category "$CATEGORY" \
    --mirror_dir "$MIRROR" --output_dir "$TMP" >/dev/null 2>&1
  PYTHON=$PYTHON "$SCRIPTS/visualize_dataset.sh" input "$MIRROR" \
    "$OUT/e2e_dataset_${SHORT}_input.html" "$K" val "$CATEGORY"
  PYTHON=$PYTHON "$SCRIPTS/visualize_dataset.sh" output "$TMP" \
    "$OUT/e2e_dataset_${SHORT}_output.html" "$K" val "$CATEGORY"
done

"$PYTHON" convert_strideqa.py \
  --source bench --mirror_dir "$MIRROR" --output_dir "$TMP" >/dev/null 2>&1
PYTHON=$PYTHON "$SCRIPTS/visualize_bench.sh" input "$MIRROR" \
  "$OUT/e2e_bench_input.html" "$K"
PYTHON=$PYTHON "$SCRIPTS/visualize_bench.sh" output "$TMP" \
  "$OUT/e2e_bench_output.html" "$K"

echo
echo "wrote:"
ls -1 "$OUT"/e2e_*.html | sed 's/^/  /'
