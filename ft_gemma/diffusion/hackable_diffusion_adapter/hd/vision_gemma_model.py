"""DiffusionGemma model with vision inputs (add-visual-inputs design).

Subclasses ``DiffusionGemma_26B_A4B`` and overrides the pieces the design
marks [DESIGN] on the transformer side (docs/add-visual-inputs/index.html):

1. ``text_only=False`` by default, so the vision tower and mm projection stay
   in the param tree (§1, config-map row ``text_only=False``).

2. ``__call__`` accepts an external ``sliding_attention_mask`` sized to the
   cache and **bypasses** ``remove_mm_logits`` (§0 "remove_mm_logits must be
   bypassed", "External sliding mask"): logits stay at full length
   ``[B, FS, V]`` aligned with the expanded sequence; the image span is
   instead masked out of the AR loss by the data pipeline.

3. ``_merge_mm_embeddings`` / ``_encode_vision`` run the vision tower
   **batched** (§0 "Batched vision encode"): ``patches [B, P_p, p_d]`` →
   ``[B, S_max, D_v]`` + validity mask, projected to ``[B, S_max, D]`` and
   scattered per example at the ``-2`` placeholder slots. The
   excess-rows-to-slot-0 semantics of ``merge_flat_embeddings`` make the
   padding rows (S_v..S_max-1) vanish, so per-example variable soft-token
   counts need no static counts and no per-batch recompilation.

Images enter only through the encoder/prefill pass; the denoiser passes
(``call_with_self_conditioning``) are byte-for-byte the baseline ones.
"""

from typing import Any

import jax.numpy as jnp

from gemma.diffusion import _models as gemma_diffusion
from gemma.gm.nn.gemma4 import _transformer as gemma4_transformer
from gemma.gm.nn.gemma4.vision import _encoder as gemma4_vision
from gemma.gm.utils import _dtype_params
from gemma.gm.vision import _token_utils

# The image, preprocessed host-side: (patches [B, P_p, p_d] f32,
# positions_xy [B, P_p, 2] int32, -1 on padding patches).
VisionInput = tuple[Any, Any]


class VisionDiffusionGemma_26B_A4B(gemma_diffusion.DiffusionGemma_26B_A4B):  # pylint: disable=invalid-name
  """DiffusionGemma 26B_A4B with the vision input path enabled."""

  # Keep config.vision_encoder + mm projection in the param tree
  # (_gemma4.py deletes them at __post_init__ when text_only=True).
  text_only: bool = False

  def __call__(  # pytype: disable=signature-mismatch
      self,
      tokens,  # Int[B, L]
      *,
      images: VisionInput | None = None,
      positions=None,  # Int[B, L]
      cache=None,
      attention_mask=None,  # Bool[B, L, cache_length]
      sliding_attention_mask=None,  # Bool[B, L, cache_length]
      return_last_only: bool | None = None,
      return_hidden_states: bool | None = None,
  ) -> gemma4_transformer.Output:
    """Transformer forward pass with batched vision inputs.

    Mirrors ``gemma4.Transformer.__call__`` with three deltas from the
    design: ``images`` is the batched ``(patches, positions_xy)`` pair,
    ``sliding_attention_mask`` may be supplied externally (sized to the
    cache), and ``remove_mm_logits`` is bypassed so the returned logits stay
    at full expanded length ``[B, L, V]``.

    Args:
      tokens: Input tokens ``[B, L]``; image soft-token slots hold ``-2``.
      images: Optional ``(patches [B, P_p, p_d], positions_xy [B, P_p, 2])``.
      positions: Input absolute positions ``[B, L]``.
      cache: Attention KV cache or None.
      attention_mask: Mask ``[B, L, cache_length]`` for the GLOBAL layers.
      sliding_attention_mask: Optional mask ``[B, L, cache_length]`` for the
        LOCAL_SLIDING layers (bidirectional within the image span). When
        None, the stock internal ``[B, L, L]`` variant is used (correct only
        when cache_length == L).
      return_last_only: If True, only return the last token's logits.
      return_hidden_states: If True, also return the hidden states.

    Returns:
      ``Output(logits [B, L, V], cache, hidden_states)``.
    """
    return_last_only = self._get_return_last_only(return_last_only)

    with _dtype_params.initialize_param_with_dtype(
        self.dtype,
        exclude=[
            # The multi-modal params are kept in float32.
            'vision_encoder',
            'embedder.mm_input_projection',
            'embedder.mm_pre_projection_norm',
            'audio_encoder',
            'embedder.audio_input_projection',
            'embedder.audio_soft_embedding_norm',
            # Skip the LoRA params
            'lora',
        ],
    ):
      inputs = self._encode_and_get_inputs(
          tokens=tokens,
          images=images,
          positions=positions,
          attention_mask=attention_mask,
      )
      del positions, attention_mask

      if sliding_attention_mask is not None:
        inputs = inputs.replace(sliding_attention_mask=sliding_attention_mask)

      x, new_cache = self._apply_attention(inputs, cache)

    if return_last_only:
      last_input_token_idx = jnp.sum(inputs.inputs_mask, axis=-1) - 1
      x = x[jnp.arange(len(x)), last_input_token_idx, ...]
    # [DESIGN] remove_mm_logits is bypassed: the stock images-path shrinks the
    # logits back to the unexpanded length using Gemma3 special tokens and a
    # fixed per-image count — wrong for gemma4 variable counts and for
    # pre-expanded prompts. Logits stay [B, L, V]; the whole expanded image
    # span is masked out of the AR loss instead (design §10 steps 4 and 10).

    logits = self.embedder.decode(x)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return gemma4_transformer.Output(
        logits=logits,
        cache=None if cache is None else new_cache,
        hidden_states=x if return_hidden_states else None,
    )

  def _merge_mm_embeddings(self, *, tokens, embeddings, images: VisionInput):
    """Merges the batched soft tokens into the text embeddings.

    Overrides the stock (batch-1, static-counts) path: the soft tokens are
    computed batched and padded to ``S_max``; ``merge_flat_embeddings``
    scatters them at the ``tokens == -2`` slots per example — padding rows
    land on slot 0 which is then restored (measured in the design's
    vision_merge_semantics.json), so variable per-example counts are safe.

    Args:
      tokens: Token ids ``[B, L]`` with ``-2`` at the soft-token slots.
      embeddings: Text embeddings ``[B, L, D]`` (garbage at the -2 slots).
      images: ``(patches [B, P_p, p_d], positions_xy [B, P_p, 2])``.

    Returns:
      Merged embeddings ``[B, L, D]``; only the ``-2`` slots changed.
    """
    soft_embeddings = self._encode_vision(images)  # [B, S_max, D]
    mask = tokens == gemma4_vision.TOKEN_PLACEHOLDER

    merged_embeddings = _token_utils.merge_flat_embeddings(
        text_embeddings=embeddings,
        multimodal_embeddings=soft_embeddings,
        mask=mask,
    )

    return merged_embeddings

  def _encode_vision(self, vision_input: VisionInput):
    """Encodes images into the text embedding space, batched.

    The tower output is padded to ``S_max`` slots per example with the first
    S_v rows valid (position-based 3x3 pooling emits valid slots first, in
    raster order); padding rows are projected too and later discarded by the
    merge.

    Args:
      vision_input: ``(patches [B, P_p, p_d], positions_xy [B, P_p, 2])``.

    Returns:
      Projected soft tokens ``[B, S_max, D]`` float32, S_v valid per example.
    """
    assert self.vision_encoder is not None

    patches, positions_xy = vision_input
    encoder_outputs = self.vision_encoder(patches, positions_xy)
    soft_tokens, _ = encoder_outputs[0]  # [B, S_max, D_v], mask [B, S_max]

    return self.embedder.encode_vision(soft_tokens)  # [B, S_max, D]
