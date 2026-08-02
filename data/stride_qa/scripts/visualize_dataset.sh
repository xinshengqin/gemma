#!/usr/bin/env bash
# Render k Dataset examples from one end of the pipeline to a standalone
# HTML page: raw + region-overlay image pairs beside the text fields.
#
# Usage:
#   scripts/visualize_dataset.sh <input|output> <dir> <output.html> \
#       [k] [split] [category]
#
#   input|output  which pipeline end: the raw mirror, or the converted Bagz
#   dir           mirror directory (input) or output directory (output)
#   k             examples to sample (default 5)
#   split         val or train (default val)
#   category      annotation category (default ego_centric_spatial_qa)
#
# Nothing is downloaded: the input end samples only records whose images are
# already mirrored, and the output end reads the Bagz via its sidecar.
#
# Example:
#   scripts/visualize_dataset.sh output out /tmp/out.html 5 val \
#       ego_centric_spatiotemporal_qa
set -euo pipefail

if [[ $# -lt 3 ]]; then
  sed -n '2,19p' "$0" >&2
  exit 64
fi

STAGE=$1
DIR=$2
HTML=$3
K=${4:-5}
SPLIT=${5:-val}
CATEGORY=${6:-ego_centric_spatial_qa}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}

# Resolve before cd, so a relative dir is read from the caller's cwd.
DIR=$(cd -- "$DIR" && pwd) || {
  echo "no such directory: $2" >&2
  exit 66
}

cd -- "$HERE"
case $STAGE in
  input)
    exec "$PYTHON" inspect_strideqa.py \
      --stage input --source dataset --mirror_dir "$DIR" \
      --split "$SPLIT" --category "$CATEGORY" --k "$K" --html "$HTML"
    ;;
  output)
    exec "$PYTHON" inspect_strideqa.py \
      --stage output --source dataset --output_dir "$DIR" \
      --split "$SPLIT" --category "$CATEGORY" --k "$K" --html "$HTML"
    ;;
  *)
    echo "stage must be 'input' or 'output', got: $STAGE" >&2
    exit 64
    ;;
esac
