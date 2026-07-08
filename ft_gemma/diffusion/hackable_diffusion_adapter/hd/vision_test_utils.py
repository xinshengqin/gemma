"""Shared fixtures for the ft_gemma vision unit tests.

Builds the tiny model from ``configs/sft_sudoku_vision_full_tiny.py`` and one
fake visual-sudoku example, folded through the exact data-pipeline
transforms of ``make_sudoku_vision_ds`` (parse -> format -> tokenize ->
patchify -> expand placeholder -> pad -> chunk -> shift).
"""

import functools

from ft_gemma.diffusion.hackable_diffusion_adapter import compat
from ft_gemma.diffusion.hackable_diffusion_adapter.data.sudoku import (
    convert_sudoku_vision,
)
from ft_gemma.diffusion.hackable_diffusion_adapter.data.sudoku import (
    sudoku_vision_data,
)
import jax
import numpy as np

compat.patch_etils_jax_prng()

# Tiny-config sequence geometry (== the design's full-config geometry).
PROMPT_LEN = 384
CANVAS_SIZE = 256
NUM_CANVASES = 1
TOTAL_CANVAS_LEN = NUM_CANVASES * CANVAS_SIZE
FULL_SEQ_LEN = PROMPT_LEN + TOTAL_CANVAS_LEN
MAX_SOFT_TOKENS = 280


@functools.cache
def make_tiny_model():
  """Resolves the tiny config and returns its ``VisionSFTDiffusion`` model."""
  from ft_gemma.diffusion.hackable_diffusion_adapter.configs import (  # pylint: disable=g-import-not-at-top
      sft_sudoku_vision_full_tiny,
  )
  from kauldron import konfig  # pylint: disable=g-import-not-at-top

  cfg = sft_sudoku_vision_full_tiny.get_config()
  return konfig.resolve(cfg.model)


@functools.cache
def make_fake_batch(seed: int = 0, batch_size: int = 1):
  """Builds a batch from one fake sudoku example via the real data pipeline.

  Args:
    seed: Seed for the fake (puzzle, solution) pair.
    batch_size: The single example is tiled to this batch size.

  Returns:
    Dict of NumPy arrays with a leading batch dimension: prompt [B, P],
    patches [B, P_p, p_d], positions_xy [B, P_p, 2], canvas [B, TC, 1],
    canvas_id [B, TC], canvas_mask [B, TC], encoder_target [B, FS],
    encoder_target_mask [B, FS].
  """
  puzzle, solution = convert_sudoku_vision.make_fake_examples(1, seed=seed)[0]
  record = convert_sudoku_vision.make_vision_record(puzzle, solution)

  # The Bagz config's transform list IS the pipeline — fold the record
  # through it (all entries are grain MapTransforms).
  ds = sudoku_vision_data.make_sudoku_vision_ds(
      bagz_path="unused.bagz",
      training=True,
      batch_size=batch_size,
      prompt_len=PROMPT_LEN,
      num_canvases=NUM_CANVASES,
      canvas_size=CANVAS_SIZE,
      max_soft_tokens_per_image=MAX_SOFT_TOKENS,
      num_workers=0,
  )
  features = record
  with jax.transfer_guard("allow"):
    for transform in ds.transforms:
      features = transform.map(features)

  return {
      key: np.broadcast_to(
          np.asarray(value)[None], (batch_size,) + np.shape(value)
      ).copy()
      for key, value in features.items()
  }


def model_kwargs_from_batch(batch):
  """Maps batch fields to ``VisionSFTDiffusion.__call__`` kwargs."""
  return dict(
      x0=batch["canvas"],
      prompt=batch["prompt"],
      canvas_id=batch["canvas_id"],
      canvas_mask=batch["canvas_mask"],
      encoder_target=batch["encoder_target"],
      encoder_target_mask=batch["encoder_target_mask"],
      patches=batch["patches"],
      positions_xy=batch["positions_xy"],
  )
