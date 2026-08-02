#!/usr/bin/env bash
# Set up this directory on a fresh clone: create a virtualenv, install the
# dependencies, run the offline golden tests, then fetch and convert a toy
# slice of both sources and render both ends to HTML.
#
# Usage:
#   scripts/setup.sh
#
# The toy run needs the network. Dataset images are extracted from the remote
# tars with HTTP range requests, so cost tracks records selected, not the
# 567 GB source; the first run also walks the tar headers of every shard it
# touches (~5 min per shard, cached in mirror/ forever after).
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-$HERE/.venv}

cd -- "$HERE"

if [[ ! -d $VENV ]]; then
  echo "==> creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi
PY=$VENV/bin/python

echo "==> installing dependencies"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet \
  absl-py bagz numpy pillow tensorflow pycocotools

echo "==> checking imports"
"$PY" - <<'EOF'
import absl, bagz, numpy, PIL, pycocotools, tensorflow
print("    ok:", ", ".join(
    f"{m.__name__} {getattr(m, '__version__', '?')}"
    for m in (numpy, tensorflow)))
EOF

echo "==> running the golden tests (offline, uses only testdata/)"
"$PY" convert_strideqa_test.py 2>&1 | tail -3

echo "==> toy conversion of the Dataset (val, 3 categories x 10 QA pairs)"
PYTHON=$PY ./scripts/run_dataset.sh

echo "==> toy conversion of the Bench (2 scene groups, 13 partitions)"
PYTHON=$PY ./scripts/run_bench.sh

cat <<EOF

Done.

  mirror/   raw source cache      $(du -sh mirror 2>/dev/null | cut -f1)
  out/      converted Bagz        $(du -sh out 2>/dev/null | cut -f1)

  open out/*.html to see both ends of both sources.
  activate the environment with: source $VENV/bin/activate
EOF
