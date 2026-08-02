#!/usr/bin/env bash
# End-to-end toy run of the Dataset source: convert every (split, category)
# combo at toy scale, then render both pipeline ends of each to HTML.
#
# Usage:
#   scripts/run_dataset.sh [max_records] [split] [category ...]
#
#   max_records   QA-pair records per category (default 10; -1 = everything
#                 in the mirrored annotation shards)
#   split         val or train (default val; see the shard-0 sampling bias
#                 note in doc/decisions_made.md before using train)
#   category      one or more categories (default: all three)
#
# Images are range-extracted from the remote tars: cost tracks the records
# selected, never the 567 GB source.
set -euo pipefail

MAX_RECORDS=${1:-10}
SPLIT=${2:-val}
if [[ $# -gt 2 ]]; then
  CATEGORIES=("${@:3}")
else
  CATEGORIES=(ego_centric_spatial_qa ego_centric_spatiotemporal_qa
              object_centric_spatial_qa)
fi
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
SCRIPTS=$HERE/scripts

cd -- "$HERE"

for CATEGORY in "${CATEGORIES[@]}"; do
  echo "==> converting $SPLIT/$CATEGORY ($MAX_RECORDS records)"
  "$PYTHON" convert_strideqa.py \
    --source dataset --split "$SPLIT" --category "$CATEGORY" \
    --max_records "$MAX_RECORDS" --mirror_dir mirror --output_dir out

  PYTHON=$PYTHON "$SCRIPTS/visualize_dataset.sh" \
    input mirror "out/dataset_${SPLIT}_${CATEGORY}_input.html" 5 \
    "$SPLIT" "$CATEGORY"
  PYTHON=$PYTHON "$SCRIPTS/visualize_dataset.sh" \
    output out "out/dataset_${SPLIT}_${CATEGORY}_output.html" 5 \
    "$SPLIT" "$CATEGORY"
done
