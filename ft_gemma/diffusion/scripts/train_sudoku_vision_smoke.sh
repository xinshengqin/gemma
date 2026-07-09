#!/bin/bash
# Smoke test for the visual-Sudoku DiffusionGemma SFT stack (add-visual-inputs
# design). Uses configs/sft_sudoku_vision_full_tiny.py (a super tiny model,
# CPU-friendly) to:
#   1. generate a tiny fake visual-sudoku Bagz dataset (if missing),
#   2. train for ONE step with the existing Kauldron trainer — forward pass,
#      backward pass, and one optimizer update,
#   3. verify every tensor shape and the attention-mask contents against the
#      design walkthrough (docs/add-visual-inputs/index.html).
#
# Usage: ft_gemma/diffusion/scripts/train_sudoku_vision_smoke.sh [workdir]

set -e

# Repo root = the directory containing ft_gemma/ (and the gemma package).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." &> /dev/null && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python3}"
for venv_dir in "${ROOT_DIR}/.venv" "${ROOT_DIR}/../.venv"; do
  if [[ -x "${venv_dir}/bin/python" ]]; then
    PYTHON="${venv_dir}/bin/python"
    break
  fi
done

# ft_gemma must be importable; the gemma repo is already installed (.pth).
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

WORKDIR="${1:-$(mktemp -d "${TMPDIR:-/tmp}/ft_gemma_sudoku_vision_smoke.XXXXXX")}"

# 1. Generate the fake visual-sudoku Bagz dataset (smoke data) if missing.
SMOKE_DATA_DIR="ft_gemma/diffusion/hackable_diffusion_adapter/data/sudoku/smoke"
if [[ ! -f "${SMOKE_DATA_DIR}/sudoku_vision_train.bagz" ]]; then
  echo "Generating fake visual-sudoku smoke data in ${SMOKE_DATA_DIR}..."
  "${PYTHON}" -m ft_gemma.diffusion.hackable_diffusion_adapter.data.sudoku.convert_sudoku_vision \
    --fake_records=8 \
    --train_split=0.75 \
    --output_dir="${SMOKE_DATA_DIR}"
fi

# 2 + 3. One-step training (fwd + bwd + optimizer update) and design
# shape/mask verification.
"${PYTHON}" -m ft_gemma.diffusion.scripts.run_train_smoke \
  --workdir="${WORKDIR}"
