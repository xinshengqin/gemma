"""Vision data transforms for DiffusionGemma SFT (add-visual-inputs design).

Host-side (NumPy, pre-jit) grain transforms from design §4:

* ``PreprocessAndPatchifyImage`` — aspect-ratio-preserving resize to
  multiples of ``pooling_kernel_size * patch_size`` px with
  ``(H_r/16)*(W_r/16) <= P_p``, normalize to [0,1] f32, patchify into 16x16
  patches and pad to ``P_p`` (reuses ``gemma4/vision/_preprocessing.py`` and
  ``gemma4/vision/_encoder.patchify_and_pad`` verbatim). Also computes
  ``S_v = num_real_patches / pooling_kernel_size**2`` for the expansion step.

* ``ExpandImagePlaceholders`` — replaces each ``<|image|>`` token (a single
  reserved id) with ``\n\n <start_of_image> [-2] x S_v <end_of_image> \n\n``
  (reuses ``_token_utils.add_variable_extra_tokens_for_images``, the same
  helper ``Gemma4Sampler`` uses at inference). The ``-2`` sentinel
  (``SOFT_TOKEN_PLACEHOLDER``) is non-PAD, so every downstream validity mask
  treats the image span as real tokens with no code change.

* ``VisionSequenceTargetShift`` — baseline ``SequenceTargetShift`` plus:
  zeroes ``encoder_target_mask`` wherever the current OR next position lies
  anywhere inside the expanded image span (soft slots AND the
  ``<soi>/<eoi>/\n\n`` markers) — the ``-2`` sentinels are unpredictable and
  the boundary markers are masked too, matching how Gemma multimodal
  fine-tuning treats image markup.
"""

from __future__ import annotations

import dataclasses

from gemma.diffusion.hackable_diffusion_adapter.data import data as adapter_data
from gemma.gm.nn.gemma4.vision import _encoder as gemma4_vision_encoder
from gemma.gm.nn.gemma4.vision import _preprocessing
from gemma.gm.vision import _token_utils
from grain import python as grain
import jax
import numpy as np

SOFT_TOKEN_PLACEHOLDER = _token_utils.SOFT_TOKEN_PLACEHOLDER

# Number of extra tokens around the soft slots: \n\n <soi> ... <eoi> \n\n.
# The span in the prompt is S_v + 4 tokens.
_NUM_MARKER_TOKENS_BEFORE = 2  # \n\n <start_of_image>
_NUM_MARKER_TOKENS_AFTER = 2  # <end_of_image> \n\n


@dataclasses.dataclass(kw_only=True, frozen=True)
class PreprocessAndPatchifyImage(grain.MapTransform):
  """Resize + normalize + patchify + pad one image, host-side.

  Reads a decoded ``uint8[H, W, 3]`` image and produces:
    - patches: ``f32[P_p, p_d]`` with ``P_p = max_soft_tokens * 3**2`` and
      ``p_d = patch_size**2 * 3``.
    - positions_xy: ``int32[P_p, 2]`` patch grid positions, -1 on padding.
    - soft_token_count: ``S_v`` — the number of valid soft tokens this image
      produces after 3x3 pooling (consumed by ``ExpandImagePlaceholders``).
  """

  image_key: str = "image"
  out_patches: str = "patches"
  out_positions_xy: str = "positions_xy"
  out_soft_token_count: str = "soft_token_count"

  patch_size: int = 16
  max_soft_tokens: int = 280
  pooling_kernel_size: int = 3

  def map(self, features):
    image = features.pop(self.image_key)
    # The reused gemma4 helpers build (host-side) jnp arrays internally;
    # kauldron guards device transfers inside the data pipeline, so allow
    # them locally. Outputs are converted back to NumPy below.
    with jax.transfer_guard("allow"):
      processed = _preprocessing.preprocess_image(
          np.asarray(image),
          patch_size=self.patch_size,
          max_soft_tokens=self.max_soft_tokens,
          pooling_kernel_size=self.pooling_kernel_size,
      )
      patches, positions_xy, num_real = (
          gemma4_vision_encoder.patchify_and_pad(
              [processed],
              patch_size=self.patch_size,
              max_soft_tokens=self.max_soft_tokens,
              pooling_kernel_size=self.pooling_kernel_size,
          )
      )
    features[self.out_patches] = np.asarray(patches[0], dtype=np.float32)
    features[self.out_positions_xy] = np.asarray(
        positions_xy[0], dtype=np.int32
    )
    features[self.out_soft_token_count] = int(num_real[0]) // (
        self.pooling_kernel_size**2
    )
    return features


@dataclasses.dataclass(kw_only=True, frozen=True)
class ExpandImagePlaceholders(grain.MapTransform):
  """Expand each ``<|image|>`` token into its S_v+4-token span.

  ``<|image|>`` (one reserved id) becomes
  ``[\n\n, <start_of_image>, -2 x S_v, <end_of_image>, \n\n]`` — net
  ``+S_v+3`` tokens. Runs at the NumPy level (pre-jit) because different
  images have different expansion sizes.
  """

  key: str = "prompt"
  soft_token_count_key: str = "soft_token_count"

  def map(self, features):
    tokens = np.asarray(features[self.key], dtype=np.int32)[None, :]
    expanded = _token_utils.add_variable_extra_tokens_for_images(
        tokens,
        soft_token_counts=[int(features[self.soft_token_count_key])],
    )
    features[self.key] = expanded[0]
    return features


def image_span_mask(prompt: np.ndarray) -> np.ndarray:
  """Marks the whole expanded image span(s) in a 1-D prompt.

  The span is the contiguous run of ``-2`` soft-token slots extended by the
  two marker tokens on each side (``\n\n <soi>`` before, ``<eoi> \n\n``
  after).

  Args:
    prompt: 1-D int array of prompt tokens.

  Returns:
    Boolean array of the same length, True inside the image span(s).
  """
  span = np.zeros(prompt.shape[0], dtype=np.bool_)
  soft = prompt == SOFT_TOKEN_PLACEHOLDER
  (indices,) = np.nonzero(soft)
  if indices.size == 0:
    return span
  # Split into contiguous runs (multiple images -> multiple runs).
  breaks = np.nonzero(np.diff(indices) > 1)[0]
  starts = np.concatenate([[indices[0]], indices[breaks + 1]])
  ends = np.concatenate([indices[breaks], [indices[-1]]])
  for start, end in zip(starts, ends):
    span_start = max(int(start) - _NUM_MARKER_TOKENS_BEFORE, 0)
    span_end = min(int(end) + _NUM_MARKER_TOKENS_AFTER, prompt.shape[0] - 1)
    span[span_start : span_end + 1] = True
  return span


@dataclasses.dataclass(kw_only=True, frozen=True)
class VisionSequenceTargetShift(adapter_data.SequenceTargetShift):
  """SequenceTargetShift that excludes the image span from the AR loss.

  Same left-shift construction of AR targets over ``prompt ++ canvas``;
  additionally zeroes ``encoder_target_mask`` wherever the current OR next
  position lies anywhere in the expanded image span
  (``\n\n <soi> [-2] x S_v <eoi> \n\n``): the ``-2`` sentinels are genuinely
  unpredictable, and the boundary markers are masked too. Text before the
  span and after the trailing ``\n\n`` stays supervised.
  """

  def map(self, features):
    features = super().map(features)

    prompt = np.asarray(features[self.in_prompt]).flatten()
    canvas = np.asarray(features[self.in_canvas]).flatten()

    span = np.concatenate(
        [image_span_mask(prompt), np.zeros(canvas.shape[0], dtype=np.bool_)]
    )
    # Position i predicts token i+1: mask out i if the current (i) or next
    # (i+1) position lies inside the image span.
    next_in_span = np.concatenate([span[1:], [False]])
    keep = ~(span | next_in_span)

    features[self.out_encoder_target_mask] = (
        np.asarray(features[self.out_encoder_target_mask]) & keep
    )
    return features
