#!/usr/bin/env bash
# Set up this directory on a fresh clone: create a virtualenv, install the
# dependencies, then fetch a small slice of the source and convert it.
#
# Usage:
#   scripts/setup.sh [max_records] [split]
#
#   max_records   how many records to fetch and convert (default 200; -1 = all)
#   split         test or train (default test)
#
# Downloads are per-label tarballs, so cost tracks the number of labels touched:
# roughly 200 MB for 200 test records, ~4 GB for 200 train records, ~73 GB for
# everything. The mirror is a cache, so nothing is ever fetched twice.
set -euo pipefail

MAX_RECORDS=${1:-200}
SPLIT=${2:-test}
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
  absl-py bagz numpy pandas pyarrow pillow tensorflow

echo "==> checking imports"
"$PY" - <<'EOF'
import absl, bagz, numpy, pandas, pyarrow, PIL, tensorflow
print("    ok:", ", ".join(
    f"{m.__name__} {getattr(m, '__version__', '?')}"
    for m in (numpy, pandas, pyarrow, tensorflow)))
EOF

echo "==> running the golden test (offline, uses only testdata/)"
"$PY" convert_msop765k_test.py 2>&1 | tail -3

echo "==> fetching and converting $MAX_RECORDS record(s) of the $SPLIT split"
"$PY" convert_msop765k.py \
  --split "$SPLIT" --max_records "$MAX_RECORDS" \
  --mirror_dir mirror --output_dir out

echo "==> rendering 5 examples from each end"
PYTHON=$PY ./scripts/visualize_ppl_input.sh mirror out/input_samples.html 5 "$SPLIT"
PYTHON=$PY ./scripts/visualize_ppl_output.sh \
  "out/msop765k_${SPLIT}@1.bagz" out/output_samples.html 5

cat <<EOF

Done.

  mirror/   raw source cache      $(du -sh mirror 2>/dev/null | cut -f1)
  out/      converted Bagz        $(du -sh out 2>/dev/null | cut -f1)

  open out/input_samples.html and out/output_samples.html to see both ends.
  activate the environment with: source $VENV/bin/activate
EOF
