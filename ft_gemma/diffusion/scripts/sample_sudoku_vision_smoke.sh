#!/bin/bash
# Inference/sampling smoke test for the visual-Sudoku DiffusionGemma stack
# (add-visual-inputs design) — counterpart of train_sudoku_vision_smoke.sh.
# Uses configs/sft_sudoku_vision_full_tiny.py (a super tiny model,
# CPU-friendly, freshly initialized parameters — no checkpoint needed) to:
#   1. generate the tiny fake visual-sudoku Bagz dataset (if missing),
#   2. verify the vision prompt prefill state (init_ar_state) against the
#      design — cache geometry, write cursor, canvas positions, and the
#      decoder attention-mask contents (image keys admitted, PAD hidden),
#   3. run the FULL autoregressive diffusion sampling loop (outer canvas
#      loop, inner DDIM denoising loop, stop-token truncation, finalize)
#      and verify the sampled outputs.
#
# Usage: ft_gemma/diffusion/scripts/sample_sudoku_vision_smoke.sh [T_s]
#   T_s: inner denoising steps (default 4, the design's tiny smoke value)

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

DENOISING_STEPS="${1:-4}"

# 1. Generate the fake visual-sudoku Bagz dataset (smoke data) if missing.
SMOKE_DATA_DIR="ft_gemma/diffusion/hackable_diffusion_adapter/data/sudoku/smoke"
if [[ ! -f "${SMOKE_DATA_DIR}/sudoku_vision_eval.bagz" ]]; then
  echo "Generating fake visual-sudoku smoke data in ${SMOKE_DATA_DIR}..."
  "${PYTHON}" -m ft_gemma.diffusion.hackable_diffusion_adapter.data.sudoku.convert_sudoku_vision \
    --fake_records=8 \
    --train_split=0.75 \
    --output_dir="${SMOKE_DATA_DIR}"
fi

# 2 + 3. Prefill verification and full AR sampling loop.
"${PYTHON}" -m ft_gemma.diffusion.scripts.run_sample_smoke \
  --denoising_steps="${DENOISING_STEPS}"
