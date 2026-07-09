"""Shared fixtures for the ft_gemma vision unit tests.

Two kinds of fixtures:

* Config-based (`make_tiny_model`, `make_fake_batch`): the tiny model from
  ``configs/sft_sudoku_vision_full_tiny.py`` and one fake visual-sudoku
  example, folded through the exact data-pipeline transforms of
  ``make_sudoku_vision_ds``. Uses the real vocab/tokenizer — slower, used by
  the end-to-end behavior tests.

* Standalone (`make_standalone_config` / `make_standalone_vision_model` /
  `make_grid_image`): a self-contained micro model (vocab 64, tiny vision
  budget S_max=4) with hand-built token rows and synthetic patch grids — no
  tokenizer, no dataset. Used by the fast API-level unit tests of the
  overridden methods.
"""

import functools

from ft_gemma.diffusion.hackable_diffusion_adapter import compat
from ft_gemma.diffusion.hackable_diffusion_adapter.data.sudoku import (
    convert_sudoku_vision,
)
from ft_gemma.diffusion.hackable_diffusion_adapter.data.sudoku import (
    sudoku_vision_data,
)
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_gemma_model
from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _modules
from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
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


################################################################################
# Standalone micro-model fixtures (no tokenizer / dataset).
################################################################################

# Micro vision budget: S_max = 4 soft-token slots -> P_p = 4 * 3**2 = 36
# patches per image, patch dim 16*16*3 = 768.
STANDALONE_VOCAB = 64
STANDALONE_S_MAX = 4
STANDALONE_MAX_PATCHES = STANDALONE_S_MAX * 9
PATCH_DIM = 768


def make_standalone_config() -> _config.TransformerConfig:
  """Micro Gemma4-MoE config, topologically like 26B_A4B + vision tower."""
  return _config.TransformerConfig(
      num_embed=STANDALONE_VOCAB,
      embed_dim=32,
      hidden_dim=48,
      num_heads=4,
      head_dim=8,
      num_kv_heads=2,
      final_logit_softcap=30.0,
      num_global_kv_heads=1,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      qk_norm_with_scale=True,
      attention_types=[
          _modules.AttentionType.LOCAL_SLIDING,
          _modules.AttentionType.LOCAL_SLIDING,
          _modules.AttentionType.GLOBAL,
      ],
      global_key_size=16,
      k_eq_v_global=True,
      global_rope_proportion=0.25,
      local_rope_proportion=1.0,
      sliding_window_size=1024,
      per_layer_input_dim=0,
      enable_moe=True,
      num_experts=4,
      expert_dim=16,
      top_k_experts=2,
      moe_dense_hidden_dim=48,
      vision_encoder=gemma_vision.VisionEncoder(
          d_model=16,
          num_layers=2,
          num_heads=2,
          ffw_hidden=32,
          output_length=STANDALONE_S_MAX,
          use_clipped_linears=False,
          standardize_embeddings=True,
      ),
      use_bidirectional_attention="vision",
  )


def make_standalone_vision_model() -> (
    vision_gemma_model.VisionDiffusionGemma_26B_A4B
):
  return vision_gemma_model.VisionDiffusionGemma_26B_A4B(
      config=make_standalone_config()
  )


def make_grid_image(
    num_patches_x: int,
    num_patches_y: int,
    seed: int = 0,
    max_patches: int = STANDALONE_MAX_PATCHES,
):
  """Builds one synthetic image as a padded patch grid.

  Mirrors ``patchify_and_pad`` output for one image: raster-order patches of
  an ``num_patches_x x num_patches_y`` grid, padded to ``max_patches`` with
  ``positions_xy = -1`` on padding. Both grid sides must be multiples of the
  3x3 pooling kernel; the image yields ``S_v = (x * y) / 9`` soft tokens.

  Args:
    num_patches_x: Grid width in patches (multiple of 3).
    num_patches_y: Grid height in patches (multiple of 3).
    seed: Seed for the patch pixel values.
    max_patches: Padded patch count P_p.

  Returns:
    (patches f32[max_patches, PATCH_DIM], positions_xy int32[max_patches, 2],
    soft_token_count int).
  """
  assert num_patches_x % 3 == 0 and num_patches_y % 3 == 0
  num_real = num_patches_x * num_patches_y
  assert num_real <= max_patches

  rng = np.random.RandomState(seed)
  patches = np.zeros((max_patches, PATCH_DIM), dtype=np.float32)
  patches[:num_real] = rng.uniform(0.0, 1.0, (num_real, PATCH_DIM))

  xs, ys = np.meshgrid(
      np.arange(num_patches_x), np.arange(num_patches_y), indexing="xy"
  )
  positions = np.full((max_patches, 2), -1, dtype=np.int32)
  positions[:num_real, 0] = xs.reshape(-1)
  positions[:num_real, 1] = ys.reshape(-1)

  return patches, positions, num_real // 9


def make_expanded_prompt(
    prompt_len: int, soft_token_count: int, text_ids: list[int]
):
  """Hand-builds an expanded prompt row: text ++ span ++ text ++ PAD.

  Layout mirrors the data pipeline's expansion,
  ``[text..., nn, soi, -2 x S_v, eoi, nn, text..., 0...]``, with arbitrary
  small ids standing in for the marker tokens (the model only special-cases
  PAD=0 and the -2 sentinel).

  Args:
    prompt_len: Padded prompt length P.
    soft_token_count: Number of -2 soft-token slots S_v.
    text_ids: Small token ids; first half goes before the span, second half
      after.

  Returns:
    int32[prompt_len] token row.
  """
  half = len(text_ids) // 2
  row = (
      text_ids[:half]
      + [11, 12]  # stand-ins for \n\n <start_of_image>
      + [-2] * soft_token_count
      + [13, 11]  # stand-ins for <end_of_image> \n\n
      + text_ids[half:]
  )
  assert len(row) <= prompt_len
  return np.array(row + [0] * (prompt_len - len(row)), dtype=np.int32)
