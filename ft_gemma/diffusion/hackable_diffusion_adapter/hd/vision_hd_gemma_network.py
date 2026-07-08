"""HD network wrapper with vision inputs (add-visual-inputs design).

Extends the baseline adapter (design §2, "MODIFIED" rows):

* ``prefill_kv_cache_with_encoder`` accepts ``images=`` and, when present,
  builds the vision prefill masks (causal + sliding-bidirectional variant,
  both sized to the cache) via ``vision_mask_helpers.make_vision_prefill_masks``
  and forwards everything into the encoder call.

* ``VisionWrappedDiffusionGemmaNetwork.encoder_call`` forwards ``images`` and
  ``sliding_attention_mask`` from the conditioning dict into
  ``Transformer.__call__`` (the vision tower runs there and the soft tokens
  are merged into the prompt embeddings, so the prefilled KV cache contains
  image K/V).

The denoiser ``__call__`` is inherited untouched — the denoiser only ever
sees the image as prompt K/V in the cache.
"""

from typing import Any

from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_mask_helpers
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers


def prefill_kv_cache_with_encoder(
    tokens,
    input_mask,
    init_cache_fn,
    encoder_fn,
    cache_length=None,
    images=None,
):
  """Prefills the KV cache with the encoder output, optionally with images.

  Same contract as the baseline
  ``hd_gemma_network.prefill_kv_cache_with_encoder``, plus the ``images``
  pair. With images, the sliding-layer mask is the vision variant
  (bidirectional within each image soft-token block), sized to the cache.

  Args:
    tokens: Input tokens of shape [B, S] (soft-token slots hold -2).
    input_mask: Boolean mask indicating valid (non-PAD) positions.
    init_cache_fn: Function to initialize the KV cache.
    encoder_fn: Function to run the encoder forward pass.
    cache_length: Total allocated cache capacity (defaults to sequence
      length).
    images: Optional ``(patches [B, P_p, p_d], positions_xy [B, P_p, 2])``.

  Returns:
    A tuple of (initialized and prefilled cache, encoder logits, positions,
    attention_mask).
  """
  if images is None:
    return hd_gemma_network.prefill_kv_cache_with_encoder(
        tokens,
        input_mask,
        init_cache_fn,
        encoder_fn,
        cache_length=cache_length,
    )

  batch_size, full_seq_len = tokens.shape
  if cache_length is None:
    cache_length = full_seq_len

  cache = init_cache_fn(
      batch_size=batch_size,
      cache_length=cache_length,
  )

  # Positions: cumsum of input_mask, 0-indexed. The -2 soft-token slots are
  # non-PAD, so they advance positions like normal tokens.
  positions = mask_helpers.build_positions_from_mask(input_mask)

  # Causal prefill mask (GLOBAL layers) + sliding variant bidirectional
  # within the image span (LOCAL_SLIDING layers), both [B, S, cache_length].
  attention_mask, sliding_attention_mask = (
      vision_mask_helpers.make_vision_prefill_masks(
          tokens=tokens,
          token_mask=input_mask,
          cache_length=cache_length,
      )
  )

  encoder_out = encoder_fn(
      x=tokens,
      conditioning_embeddings={
          'kv_cache': cache,
          'positions': positions,
          'attention_mask': attention_mask,
          'sliding_attention_mask': sliding_attention_mask,
          'images': images,
      },
  )
  kv_cache = encoder_out.cache
  encoder_logits = encoder_out.logits
  if kv_cache is None:
    raise ValueError('KV cache should not be None after encoder pass')
  return kv_cache, encoder_logits, positions, attention_mask


class VisionWrappedDiffusionGemmaNetwork(
    hd_gemma_network.WrappedDiffusionGemmaNetwork
):
  """Wrapped DiffusionGemma network that threads images into the encoder.

  ``encoder_call`` additionally forwards ``images`` and
  ``sliding_attention_mask`` from ``conditioning_embeddings`` into
  ``Transformer.__call__``. Without images it behaves exactly like the
  baseline wrapper (used when appending sampled canvases to the cache at
  inference). The denoiser ``__call__`` is inherited unchanged.
  """

  def encoder_call(
      self,
      *,
      x,
      conditioning_embeddings: dict[str, Any],
  ):
    """Calls the Gemma encoder, forwarding vision conditioning if present.

    Args:
      x: Input tokens or array.
      conditioning_embeddings: Dictionary containing cache, positions,
        attention_mask, and optionally images and sliding_attention_mask.

    Returns:
      The transformer output containing updated cache and logits.
    """
    images = conditioning_embeddings.get('images', None)
    sliding_attention_mask = conditioning_embeddings.get(
        'sliding_attention_mask', None
    )
    if images is None and sliding_attention_mask is None:
      return super().encoder_call(
          x=x, conditioning_embeddings=conditioning_embeddings
      )

    if len(x.shape) == 3:
      tokens = x[..., 0]
    else:
      tokens = x
    assert len(tokens.shape) == 2

    cache = conditioning_embeddings.get('kv_cache', None)
    positions = conditioning_embeddings.get('positions', None)
    attention_mask = conditioning_embeddings.get('attention_mask', None)
    return self.gemma_model(
        tokens=tokens,
        images=images,
        cache=cache,
        positions=positions,
        attention_mask=attention_mask,
        sliding_attention_mask=sliding_attention_mask,
    )
