"""Vision-aware attention-mask helpers for DiffusionGemma.

Implements ``make_vision_prefill_masks`` from the add-visual-inputs design
(docs/add-visual-inputs/index.html, §5.1 / encoder-call diagram):

* The **causal prefill mask** is unchanged from
  ``mask_helpers.make_causal_prefill_mask`` — ``mask[b, q, k] = (k <= q) AND
  valid[b, k]``, right-padded to the cache length. It is consumed by the 5
  GLOBAL attention layers (image tokens stay causal there, exactly like the
  AR gemma4).

* The **sliding variant** is additionally bidirectional within each
  contiguous image soft-token block (the ``-2`` sentinel runs), mirroring what
  the AR gemma4 builds internally for ``use_bidirectional_attention='vision'``
  (gemma/gm/utils/_attention_mask.py) — but sized to the *cache* axis so the
  same builder works at inference prefill where cache_length > seq_len (the
  stock internal builder is [B, L, L] and would mismatch there). It is
  consumed by the 25 LOCAL_SLIDING layers, which skip their sliding-window
  check when this mask is supplied.
"""

import jax.numpy as jnp

from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
from gemma.gm.utils import _attention_mask
from gemma.gm.vision import _token_utils

SOFT_TOKEN_PLACEHOLDER = _token_utils.SOFT_TOKEN_PLACEHOLDER


def make_vision_prefill_masks(
    tokens: jnp.ndarray,
    token_mask: jnp.ndarray,
    cache_length: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Builds the causal prefill mask plus its vision-bidirectional variant.

  Args:
    tokens: Token ids of shape ``[B, L]``. Image soft-token slots hold the
      ``-2`` sentinel (``SOFT_TOKEN_PLACEHOLDER``).
    token_mask: Boolean validity mask of shape ``[B, L]`` (True for real,
      non-PAD tokens; the ``-2`` slots are non-PAD, hence True).
    cache_length: Total cache width the masks are right-padded to.

  Returns:
    A tuple ``(attention_mask, sliding_attention_mask)``:
      - attention_mask: ``[B, L, cache_length]`` strictly causal mask for the
        GLOBAL layers.
      - sliding_attention_mask: ``[B, L, cache_length]`` mask for the
        LOCAL_SLIDING layers — causal everywhere except within each
        contiguous image block, whose soft tokens see each other fully.
  """
  attention_mask = mask_helpers.make_causal_prefill_mask(
      token_mask, cache_length
  )

  bidirectional_mask = tokens == SOFT_TOKEN_PLACEHOLDER
  sliding_attention_mask = (
      _attention_mask.make_causal_bidirectional_attention_mask(
          token_mask,
          bidirectional_mask=bidirectional_mask,
      )
  )  # [B, L, L]

  seq_len = tokens.shape[-1]
  pad_width = cache_length - seq_len
  if pad_width > 0:
    sliding_attention_mask = jnp.pad(
        sliding_attention_mask,
        ((0, 0), (0, 0), (0, pad_width)),
        constant_values=False,
    )

  return attention_mask, sliding_attention_mask
