"""Gemma AR state handler with vision inputs (add-visual-inputs design).

Overrides ``GemmaARStateHandler.init_ar_state`` (design §6, §11 step 2): the
prompt prefill runs one causal encoder pass with ``images=`` forwarded — the
vision tower embeds the patches, the soft tokens are merged at the ``-2``
slots, and the resulting image K/V land in the cache (``end_index = P``).
The sliding mask for this pass is the cache-length vision variant
(bidirectional within the image span) from ``make_vision_prefill_masks``.

Both sampling loops (outer AR canvas loop, inner denoising loop) and
``update_ar_state`` / ``finalize_ar_state`` are inherited untouched — the
canvas-append forward pass carries no images, so the inherited
``encoder_call`` path behaves exactly like the baseline there.
"""

import dataclasses

from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_hd_gemma_network
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_ar_state_handler
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
from hackable_diffusion.lib.sampling import ar_diffusion_sampler
import jax.numpy as jnp

Conditioning = hd_gemma_ar_state_handler.Conditioning


@dataclasses.dataclass(kw_only=True)
class VisionGemmaARStateHandler(hd_gemma_ar_state_handler.GemmaARStateHandler):
  """AR state handler whose prompt prefill embeds and merges the image."""

  def init_ar_state(
      self,
      *,
      batch_size: int,
      conditioning: Conditioning,
      canvas_length: int,
      max_num_canvases: int,
  ) -> ar_diffusion_sampler.SamplerState:
    """Creates the initial sampler state, prefetching image K/V into the cache.

    Args:
      batch_size: The batch size.
      conditioning: Initial conditioning dict containing prompt
        tokens/lengths and the image tensors (``patches``,
        ``positions_xy``).
      canvas_length: Number of tokens per AR canvas generation step.
      max_num_canvases: Maximum number of canvases that can be generated.

    Returns:
      The initial AR diffusion sampler state dict.
    """
    ##########################################################################
    # Extract pre-tokenized prompt tokens and lengths from conditioning.
    ##########################################################################
    prompt_tokens = conditioning["prompt_tokens"]
    prompt_lengths = conditioning["prompt_lengths"]
    images = (conditioning["patches"], conditioning["positions_xy"])
    max_prompt_len = prompt_tokens.shape[1]
    cache_length = max_prompt_len + max_num_canvases * canvas_length
    ##########################################################################
    # Derive batch dimensions and input mask.
    ##########################################################################
    input_mask = jnp.arange(max_prompt_len)[None, :] < prompt_lengths[:, None]

    # [DESIGN] the vision tower runs HERE and only here at inference: the
    # prefill masks are the cache-length vision variants and the image K/V
    # land in the cache (end_index = P).
    cache, _, _, _ = vision_hd_gemma_network.prefill_kv_cache_with_encoder(
        tokens=prompt_tokens,
        input_mask=input_mask,
        init_cache_fn=self.init_cache_fn,
        encoder_fn=self.encoder_fn,
        cache_length=cache_length,
        images=images,
    )

    ##########################################################################
    # Pre-compute full_attention_mask for permanent pad masking.
    ##########################################################################
    # full_attention_mask: (B, cache_length) — True for real prompt tokens
    # and all future decode slots; False for right-pad slots.
    full_attention_mask = mask_helpers.make_full_attention_mask(
        input_mask, cache_length=cache_length
    )

    ##########################################################################
    # Build canvas positions and attention mask for the first canvas.
    ##########################################################################
    # Positions: per-element, starting after each prompt's last real token.
    # (Gemma4 end_index is a write cursor = max_prompt_len for all elements,
    #  but RoPE positions must reflect actual prompt lengths.)
    canvas_positions = (
        prompt_lengths[:, None] + jnp.arange(canvas_length)[None, :]
    )

    total_canvas_len = cache_length - max_prompt_len
    # The decoder mask rule is unchanged — it admits the image keys
    # automatically, because the -2 soft-token slots are non-PAD prompt
    # positions (design §1).
    canvas_attn_mask = mask_helpers.create_decoder_attention_mask(
        prompt_mask=input_mask,
        canvas_mask=jnp.ones(
            (batch_size, total_canvas_len), dtype=jnp.bool_
        ),  # currently there are no pad tokens in our canvases to be generated
        selected_canvas_idx=jnp.zeros(
            (batch_size,), dtype=jnp.int32
        ),  # we are generating the first (0-th) canvas
        prompt_len=max_prompt_len,
        total_canvas_len=total_canvas_len,
        canvas_size=canvas_length,
        num_queries=canvas_length,
    )  # (B, canvas_length, cache_length)

    ##########################################################################
    # Allocate output buffer.
    ##########################################################################
    all_canvas_tokens = jnp.zeros(
        (batch_size, max_num_canvases * canvas_length), dtype=jnp.int32
    )
    predicted_tokens = jnp.concatenate(
        [prompt_tokens, all_canvas_tokens], axis=1
    )

    ##########################################################################
    # Assemble initial state.
    ##########################################################################
    init_ar_state = {
        "prompt_tokens": prompt_tokens,
        "prompt_lengths": prompt_lengths,
        "prompt_mask": input_mask,
        "predicted_tokens": predicted_tokens,
        "step": max_prompt_len,
        "done": jnp.zeros(shape=(batch_size,), dtype=jnp.bool_),
        "kv_cache": cache,
        "positions": canvas_positions,
        "attention_mask": canvas_attn_mask,
        "full_attention_mask": full_attention_mask,
        "processed_denoising_steps": 0,
        "processed_num_canvases": 0,
    }
    return init_ar_state
