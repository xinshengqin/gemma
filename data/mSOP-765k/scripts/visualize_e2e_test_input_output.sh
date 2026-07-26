#!/usr/bin/env bash
# Render both ends of the end-to-end test fixture, so what the golden test
# compares can be eyeballed rather than inferred from a binary.
#
# Writes, under testdata/visualization/:
#   e2e_test_input.html    the fixture mirror the test converts
#   e2e_test_output.html   testdata/golden.bagz, the expected result
#
# Usage:
#   scripts/visualize_e2e_test_input_output.sh [k]
#
# The fixture holds a single example, so k only matters if it ever grows.
set -euo pipefail

K=${1:-5}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPTS=$HERE/scripts
OUT=$HERE/testdata/visualization

mkdir -p -- "$OUT"

"$SCRIPTS/visualize_ppl_input.sh" \
  "$HERE/testdata/mirror" "$OUT/e2e_test_input.html" "$K" test

"$SCRIPTS/visualize_ppl_output.sh" \
  "$HERE/testdata/golden.bagz" "$OUT/e2e_test_output.html" "$K"

echo
echo "wrote:"
echo "  $OUT/e2e_test_input.html"
echo "  $OUT/e2e_test_output.html"
