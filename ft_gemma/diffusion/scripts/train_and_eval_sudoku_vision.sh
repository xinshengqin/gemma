#!/bin/bash
# Train + offline eval for the visual-Sudoku DiffusionGemma SFT config,
# following the launch convention of
# gemma/diffusion/hackable_diffusion_adapter/README.md:
#
#   python3 -m kauldron.main --cfg=<config>.py --cfg.workdir=...
#   python3 -m ft_gemma.diffusion.hackable_diffusion_adapter.eval_main \
#       --cfg=<config>.py --task=sudoku --cfg.workdir=...
#
# Defaults to the tiny smoke config (1 train step + 1 eval batch, CPU-sized).
# For a production run, pass the full config (and prepare the real dataset
# with data/sudoku/convert_sudoku_vision.py first):
#
#   ft_gemma/diffusion/scripts/train_and_eval_sudoku_vision.sh \
#     ft_gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku_vision_full.py \
#     $(pwd)/xp_dir_sudoku_vision
#
# Usage: train_and_eval_sudoku_vision.sh [config] [workdir] [eval_names]

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

# ft_gemma must be importable; the gemma package is already installed (.pth).
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CFG="${1:-ft_gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku_vision_full_tiny.py}"
WORKDIR="${2:-$(pwd)/xp_dir_sudoku_vision}"
EVAL_NAMES="${3:-sample_ar_steps32}"

# Training.
# Command line overrides are used to prevent compilation hangs and NCCL
# errors (see the adapter README; the NCCL settings are inert on CPU).
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    NCCL_ALGO="Ring" \
    NCCL_PROTO="LL128" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_CUMEM_ENABLE="0" \
    "${PYTHON}" -m kauldron.main \
  --cfg="${CFG}" \
  --cfg.workdir="${WORKDIR}"

# Offline evaluation on the saved checkpoint (AR diffusion sampling).
env XLA_FLAGS="--xla_disable_hlo_passes=constant_folding" \
    XLA_PYTHON_CLIENT_PREALLOCATE="false" \
    TF_FORCE_GPU_ALLOW_GROWTH="true" \
    "${PYTHON}" -m ft_gemma.diffusion.hackable_diffusion_adapter.eval_main \
    --cfg="${CFG}" \
    --task=sudoku \
    --eval_names="${EVAL_NAMES}" \
    --cfg.workdir="${WORKDIR}"
