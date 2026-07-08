"""Unit tests for the vision HD network wrapper and vision prefill.

Intended behavior:

  * ``VisionWrappedDiffusionGemmaNetwork.encoder_call`` WITHOUT images or a
    sliding mask in the conditioning is byte-identical to the baseline
    wrapper (this is the path used when appending sampled canvases to the
    cache at inference).
  * ``prefill_kv_cache_with_encoder`` with ``images=None`` delegates exactly
    to the baseline function.
  * With images, the prefill returns FULL-length logits, leaves the write
    cursor at seq_len, and writes image-derived K/V into the cache at the
    soft-token positions — that cached K/V is the only way the denoiser ever
    sees the image, so it must change when the image changes while the text
    K/V before the span stays bit-identical (causality).
  * The cache may be longer than the prompt (inference prefill geometry).
"""

from absl.testing import absltest
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import (
    vision_hd_gemma_network,
)
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_mask_helpers
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_test_utils
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
import jax
import jax.numpy as jnp
import numpy as np

P = 16
VOCAB = vision_test_utils.STANDALONE_VOCAB


class VisionHdGemmaNetworkTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.config = vision_test_utils.make_standalone_config()
    gemma_model = vision_test_utils.make_standalone_vision_model()
    cls.net = vision_hd_gemma_network.VisionWrappedDiffusionGemmaNetwork(
        gemma_model=gemma_model
    )
    # Baseline wrapper around the SAME inner model: the two wrappers share
    # the param-tree structure ({'gemma_model': ...}).
    cls.baseline_net = hd_gemma_network.WrappedDiffusionGemmaNetwork(
        gemma_model=gemma_model
    )

    patches, positions_xy, s_v = vision_test_utils.make_grid_image(6, 3, 0)
    cls.patches = jnp.asarray(patches)[None]  # [1, P_p, p_d]
    cls.positions_xy = jnp.asarray(positions_xy)[None]  # [1, P_p, 2]
    cls.s_v = s_v
    cls.tokens = jnp.asarray(
        vision_test_utils.make_expanded_prompt(P, s_v, [2, 5, 6, 7])
    )[None]  # [1, P]
    cls.input_mask = cls.tokens != 0

    # Init params through the images-enabled encoder path.
    cache = cls._make_cache(cache_length=P)
    positions = mask_helpers.build_positions_from_mask(cls.input_mask)
    attn, sliding = vision_mask_helpers.make_vision_prefill_masks(
        tokens=cls.tokens, token_mask=cls.input_mask, cache_length=P
    )
    cls.variables = cls.net.init(
        jax.random.PRNGKey(0),
        x=cls.tokens,
        conditioning_embeddings={
            'kv_cache': cache,
            'positions': positions,
            'attention_mask': attn,
            'sliding_attention_mask': sliding,
            'images': (cls.patches, cls.positions_xy),
        },
        method=cls.net.encoder_call,
    )

  @classmethod
  def _make_cache(cls, cache_length):
    return cls.config.init_cache(
        batch_size=1, dtype=jnp.bfloat16, cache_length=cache_length
    )

  def _prefill(self, images, tokens=None, cache_length=None):
    tokens = self.tokens if tokens is None else tokens
    input_mask = tokens != 0
    return vision_hd_gemma_network.prefill_kv_cache_with_encoder(
        tokens=tokens,
        input_mask=input_mask,
        init_cache_fn=lambda batch_size, cache_length: (
            self.config.init_cache(
                batch_size=batch_size,
                dtype=jnp.bfloat16,
                cache_length=cache_length,
            )
        ),
        encoder_fn=lambda x, conditioning_embeddings: self.net.apply(
            self.variables,
            x=x,
            conditioning_embeddings=conditioning_embeddings,
            method=self.net.encoder_call,
        ),
        cache_length=cache_length,
        images=images,
    )

  ##############################################################################
  # Baseline equivalence when no vision conditioning is present.
  ##############################################################################

  def test_encoder_call_without_images_matches_baseline_wrapper(self):
    """No images / no sliding mask in the conditioning -> exact baseline
    behavior (the canvas-append path at inference relies on this)."""
    text_tokens = jnp.where(self.tokens == -2, 3, self.tokens)  # plain text
    input_mask = text_tokens != 0
    positions = mask_helpers.build_positions_from_mask(input_mask)
    attn = mask_helpers.make_causal_prefill_mask(input_mask, P)
    conditioning = {
        'kv_cache': self._make_cache(P),
        'positions': positions,
        'attention_mask': attn,
    }
    out_vision = self.net.apply(
        self.variables,
        x=text_tokens,
        conditioning_embeddings=dict(conditioning),
        method=self.net.encoder_call,
    )
    out_baseline = self.baseline_net.apply(
        self.variables,
        x=text_tokens,
        conditioning_embeddings=dict(conditioning),
        method=self.baseline_net.encoder_call,
    )
    np.testing.assert_array_equal(
        np.asarray(out_vision.logits), np.asarray(out_baseline.logits)
    )
    jax.tree.map(
        lambda a, b: np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b)
        ),
        out_vision.cache,
        out_baseline.cache,
    )

  def test_prefill_without_images_matches_baseline_function(self):
    """images=None delegates to hd_gemma_network.prefill_kv_cache_with_encoder."""
    text_tokens = jnp.where(self.tokens == -2, 3, self.tokens)
    cache_v, logits_v, pos_v, mask_v = self._prefill(
        images=None, tokens=text_tokens
    )
    cache_b, logits_b, pos_b, mask_b = (
        hd_gemma_network.prefill_kv_cache_with_encoder(
            tokens=text_tokens,
            input_mask=text_tokens != 0,
            init_cache_fn=lambda batch_size, cache_length: (
                self.config.init_cache(
                    batch_size=batch_size,
                    dtype=jnp.bfloat16,
                    cache_length=cache_length,
                )
            ),
            encoder_fn=lambda x, conditioning_embeddings: self.net.apply(
                self.variables,
                x=x,
                conditioning_embeddings=conditioning_embeddings,
                method=self.net.encoder_call,
            ),
        )
    )
    np.testing.assert_array_equal(np.asarray(logits_v), np.asarray(logits_b))
    np.testing.assert_array_equal(np.asarray(pos_v), np.asarray(pos_b))
    np.testing.assert_array_equal(np.asarray(mask_v), np.asarray(mask_b))
    jax.tree.map(
        lambda a, b: np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b)
        ),
        cache_v,
        cache_b,
    )

  ##############################################################################
  # Vision prefill contract.
  ##############################################################################

  def test_prefill_kv_cache_with_encoder_canonical_usage(self):
    """One toy example; all parameter-free outputs written out.

    Input: one 6-token prompt with a 2-slot image span, an 8-slot cache,
    and one 6x3-patch image (S_v = 2):

        position:  0    1     2     3     4     5
        token:     2    5    -2    -2     6     0
                  bos  text  soft  soft  text  PAD

    (Logits and cache K/V values depend on network parameters, so this test
    pins their shapes and cursor; the design's numeric behavior is covered
    by the property tests below.)
    """
    tokens = jnp.asarray([[2, 5, -2, -2, 6, 0]], dtype=jnp.int32)
    patches, positions_xy, s_v = vision_test_utils.make_grid_image(6, 3, 0)
    self.assertEqual(s_v, 2)
    cache, logits, positions, attn = self._prefill(
        images=(jnp.asarray(patches)[None], jnp.asarray(positions_xy)[None]),
        tokens=tokens,
        cache_length=8,
    )

    # RoPE positions: cumsum(valid) - 1. The -2 soft slots are non-PAD, so
    # they advance the counter like normal tokens; the PAD repeats.
    np.testing.assert_array_equal(positions, [[0, 1, 2, 3, 4, 4]])

    # Returned (causal) attention mask over the 8-slot cache.
    np.testing.assert_array_equal(
        np.asarray(attn[0], dtype=int),
        [
            # keys:  2  5 -2 -2  6  P  .  .   (cols 6-7: unwritten cache)
            [1, 0, 0, 0, 0, 0, 0, 0],  # q0: bos
            [1, 1, 0, 0, 0, 0, 0, 0],  # q1: text
            [1, 1, 1, 0, 0, 0, 0, 0],  # q2: soft token
            [1, 1, 1, 1, 0, 0, 0, 0],  # q3: soft token
            [1, 1, 1, 1, 1, 0, 0, 0],  # q4: text
            [1, 1, 1, 1, 1, 0, 0, 0],  # q5: PAD row
        ],
    )

    # Full-length logits (remove_mm_logits bypassed) and write cursor at 6.
    self.assertEqual(logits.shape, (1, 6, VOCAB))
    np.testing.assert_array_equal(np.asarray(cache['layer_0']['end_index']), [6])
    self.assertEqual(cache['layer_0']['k'].shape, (1, 8, 2, 8))

    # Which cache slots hold REAL (written) K/V vs stay empty:
    #
    #   slot:     0    1    2     3     4     5     6  7
    #   content: bos  text soft  soft  text  PAD    -  -
    #   written:  y    y    y     y     y     y     n  n
    #
    # ALL six prompt slots are written — including the PAD at slot 5, whose
    # K/V is garbage: padding is protected by the masks above (key column 5
    # is False for every query), NOT by skipping the write. Slots 2-3 hold
    # the image-derived K/V. Slots 6-7 were never written.
    k = np.asarray(cache['layer_0']['k'], dtype=np.float32)
    for slot in (0, 1, 2, 3, 4, 5):
      self.assertTrue(np.any(k[0, slot] != 0.0), f'slot {slot} not written')
    np.testing.assert_array_equal(k[0, 6:], 0.0)

  def test_prefill_with_images_full_length_logits_and_cursor(self):
    cache, logits, positions, attn = self._prefill(
        images=(self.patches, self.positions_xy)
    )
    self.assertEqual(logits.shape, (1, P, VOCAB))  # remove_mm_logits bypassed
    self.assertEqual(attn.shape, (1, P, P))
    self.assertEqual(positions.shape, (1, P))
    np.testing.assert_array_equal(
        np.asarray(cache['layer_0']['end_index']), [P]
    )

  def test_image_kv_lands_in_cache_at_soft_positions(self):
    """Changing the image changes the cached K/V at the soft-token slots;
    text K/V BEFORE the span is untouched (causal) — the image reaches the
    denoiser only through these cache entries."""
    other_patches = jnp.asarray(
        vision_test_utils.make_grid_image(6, 3, seed=7)[0]
    )[None]
    cache_a, _, _, _ = self._prefill(images=(self.patches, self.positions_xy))
    cache_b, _, _, _ = self._prefill(images=(other_patches, self.positions_xy))

    tokens = np.asarray(self.tokens[0])
    soft = np.nonzero(tokens == -2)[0]
    span_start = int(soft[0]) - 2  # \n\n <soi> precede the slots

    for layer in cache_a:
      k_a, k_b = np.asarray(cache_a[layer]['k']), np.asarray(cache_b[layer]['k'])
      # K at positions strictly before the image span: bit-identical.
      np.testing.assert_array_equal(
          k_a[:, :span_start],
          k_b[:, :span_start],
          err_msg=f'{layer}: pre-span K must not depend on the image',
      )
      # K at the soft-token positions: must differ (image K/V in the cache).
      self.assertTrue(
          np.any(k_a[:, soft] != k_b[:, soft]),
          f'{layer}: soft-token K unchanged when the image changed',
      )

  def test_prefill_cache_longer_than_sequence(self):
    """Inference prefill geometry: cache C > P; masks sized to C; slots
    beyond the prompt stay empty and the cursor stays at P."""
    cache_length = P + 8
    cache, logits, _, attn = self._prefill(
        images=(self.patches, self.positions_xy), cache_length=cache_length
    )
    self.assertEqual(logits.shape, (1, P, VOCAB))
    self.assertEqual(attn.shape, (1, P, cache_length))
    layer0 = cache['layer_0']
    self.assertEqual(layer0['k'].shape[1], cache_length)
    np.testing.assert_array_equal(np.asarray(layer0['end_index']), [P])
    np.testing.assert_array_equal(
        np.asarray(layer0['k'][:, P:], dtype=np.float32),
        np.zeros_like(np.asarray(layer0['k'][:, P:], dtype=np.float32)),
    )


if __name__ == '__main__':
  absltest.main()
