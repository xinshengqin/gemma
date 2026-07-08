"""Unit tests for ``VisionDiffusionGemma_26B_A4B`` (the transformer override).

Intended behavior, per the design and the API contract:

  * ``text_only=False`` by default — the vision tower + mm projection stay in
    the config/param tree (the stock default deletes them at __post_init__).
  * ``images`` is the batched pair ``(patches [B, P_p, p_d], positions_xy
    [B, P_p, 2])`` — ONE image per example, arbitrary B, per-example variable
    soft-token counts within the same batch, no static counts.
  * At B=1 the batched encode+merge is numerically equivalent to the stock
    ``PreprocessedVisionInput`` path (`_encode_vision` with static
    ``soft_token_counts`` + flat merge) — the override changes the batching,
    not the math.
  * ``remove_mm_logits`` is bypassed: logits keep the full expanded length.
  * An externally supplied ``sliding_attention_mask`` (sized to the cache) is
    consumed by the LOCAL_SLIDING layers.
  * Feeding the batched tuple to a STOCK model fails fast at the typecheck —
    the two input conventions cannot be silently mixed.
"""

from absl.testing import absltest
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_mask_helpers
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_test_utils
from gemma.diffusion import _models as gemma_diffusion
from kauldron.ktyping import errors as ktyping_errors
import jax
import jax.numpy as jnp
import numpy as np

P = 16  # padded prompt length for these micro tests
S_MAX = vision_test_utils.STANDALONE_S_MAX
P_P = vision_test_utils.STANDALONE_MAX_PATCHES
VOCAB = vision_test_utils.STANDALONE_VOCAB


class VisionDiffusionGemmaTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.model = vision_test_utils.make_standalone_vision_model()

    # Two examples with DIFFERENT soft-token counts in one batch:
    #   example 0: 6x3-patch grid  -> S_v = 2
    #   example 1: 9x3-patch grid  -> S_v = 3
    patches_a, positions_a, s_v_a = vision_test_utils.make_grid_image(6, 3, 0)
    patches_b, positions_b, s_v_b = vision_test_utils.make_grid_image(9, 3, 1)
    assert (s_v_a, s_v_b) == (2, 3)
    cls.soft_counts = (s_v_a, s_v_b)
    cls.patches = jnp.stack([patches_a, patches_b])  # [2, P_p, p_d]
    cls.positions_xy = jnp.stack([positions_a, positions_b])  # [2, P_p, 2]
    cls.tokens = jnp.stack([
        jnp.asarray(
            vision_test_utils.make_expanded_prompt(P, s_v_a, [2, 5, 6, 7])
        ),
        jnp.asarray(
            vision_test_utils.make_expanded_prompt(P, s_v_b, [2, 8, 3, 4])
        ),
    ])  # [2, P]

    cls.variables = cls.model.init(
        jax.random.PRNGKey(0),
        cls.tokens,
        images=(cls.patches, cls.positions_xy),
    )
    cls.output = cls.model.apply(
        cls.variables, cls.tokens, images=(cls.patches, cls.positions_xy)
    )

  ##############################################################################
  # Param-tree / config behavior.
  ##############################################################################

  def test_text_only_false_keeps_vision_tower(self):
    """Default text_only=False keeps tower + mm projection; stock deletes."""
    self.assertFalse(self.model.text_only)
    self.assertIsNotNone(self.model.config.vision_encoder)
    params = self.variables['params']
    self.assertIn('vision_encoder', params)
    self.assertIn('mm_input_projection', params['embedder'])

    # The stock class with the same config but its default text_only=True
    # strips the tower from the config at construction time.
    stock = gemma_diffusion.DiffusionGemma_26B_A4B(
        config=vision_test_utils.make_standalone_config()
    )
    self.assertTrue(stock.text_only)
    self.assertIsNone(stock.config.vision_encoder)

  ##############################################################################
  # Batched images input: B > 1, one image per example, variable S_v.
  ##############################################################################

  def test_batched_rows_match_individual_runs(self):
    """Row b of a B=2 run == the B=1 run of example b (variable S_v).

    This pins the batched-input contract: examples are independent along the
    batch axis even when their soft-token counts differ — the property the
    stock static-counts path cannot provide.
    """
    for b in range(2):
      single = self.model.apply(
          self.variables,
          self.tokens[b : b + 1],
          images=(self.patches[b : b + 1], self.positions_xy[b : b + 1]),
      )
      np.testing.assert_allclose(
          np.asarray(self.output.logits[b], dtype=np.float32),
          np.asarray(single.logits[0], dtype=np.float32),
          atol=1e-5,
          err_msg=f'batched logits row {b} != individual run of example {b}',
      )

  def test_batched_merge_matches_stock_batch1_path(self):
    """At B=1 the batched encode+merge == the stock PreprocessedVisionInput
    path, example by example — same math, different batching."""
    from gemma.gm.nn.gemma4 import _transformer as gemma4_transformer

    # A stock (vision-enabled) model sharing the same parameters: only the
    # class differs, so methods resolve to the stock implementations.
    stock = gemma_diffusion.DiffusionGemma_26B_A4B(
        config=vision_test_utils.make_standalone_config(), text_only=False
    )

    def _embed(model, tokens):
      return model.apply(
          self.variables,
          tokens,
          method=lambda m, t: m.embedder.encode(t),
      )

    for b in range(2):
      tokens = self.tokens[b : b + 1]
      embeddings = _embed(self.model, tokens)

      merged_batched = self.model.apply(
          self.variables,
          tokens=tokens,
          embeddings=embeddings,
          images=(self.patches[b : b + 1], self.positions_xy[b : b + 1]),
          method='_merge_mm_embeddings',
      )

      # Stock packing: [1, n_images * max_patches, p_d] + static counts.
      stock_images = gemma4_transformer.PreprocessedVisionInput(
          patches=self.patches[b : b + 1].reshape(1, P_P, -1),
          positions_xy=self.positions_xy[b : b + 1].reshape(1, P_P, 2),
          soft_token_counts=(self.soft_counts[b],),
      )
      merged_stock = stock.apply(
          self.variables,
          tokens=tokens,
          embeddings=embeddings,
          images=stock_images,
          method='_merge_mm_embeddings',
      )

      np.testing.assert_allclose(
          np.asarray(merged_batched, dtype=np.float32),
          np.asarray(merged_stock, dtype=np.float32),
          atol=1e-6,
          err_msg=(
              f'example {b}: batched merge differs from the stock'
              ' static-counts path at B=1'
          ),
      )

  def test_encode_vision_is_batched_and_padded_to_s_max(self):
    """_encode_vision returns [B, S_max, D]: padded, no static counts."""
    soft = self.model.apply(
        self.variables,
        (self.patches, self.positions_xy),
        method='_encode_vision',
    )
    self.assertEqual(
        soft.shape, (2, S_MAX, self.model.config.embed_dim)
    )

  def test_merge_changes_only_the_soft_token_slots(self):
    """Merged embeddings differ from text embeddings exactly at the -2 slots
    (padding soft rows are discarded via the slot-0 restore)."""
    embeddings = self.model.apply(
        self.variables,
        self.tokens,
        method=lambda m, t: m.embedder.encode(t),
    )
    merged = self.model.apply(
        self.variables,
        tokens=self.tokens,
        embeddings=embeddings,
        images=(self.patches, self.positions_xy),
        method='_merge_mm_embeddings',
    )
    changed = ~np.all(
        np.asarray(merged) == np.asarray(embeddings), axis=-1
    )  # [B, P]
    expected = np.asarray(self.tokens) == -2
    np.testing.assert_array_equal(changed, expected)

  ##############################################################################
  # remove_mm_logits bypass.
  ##############################################################################

  def test_logits_keep_full_expanded_length(self):
    """[DESIGN] remove_mm_logits bypassed: logits stay [B, L, V], aligned
    with the expanded sequence (the AR loss masks the image span instead)."""
    self.assertEqual(self.output.logits.shape, (2, P, VOCAB))

  ##############################################################################
  # External sliding mask.
  ##############################################################################

  def test_external_sliding_mask_is_consumed(self):
    """A causal-only sliding mask changes the result vs the vision one —
    proof the LOCAL_SLIDING layers consume the supplied mask."""
    token_mask = self.tokens != 0
    attn, sliding = vision_mask_helpers.make_vision_prefill_masks(
        tokens=self.tokens, token_mask=token_mask, cache_length=P
    )
    out_bidir = self.model.apply(
        self.variables,
        self.tokens,
        images=(self.patches, self.positions_xy),
        attention_mask=attn,
        sliding_attention_mask=sliding,
    )
    out_causal = self.model.apply(
        self.variables,
        self.tokens,
        images=(self.patches, self.positions_xy),
        attention_mask=attn,
        sliding_attention_mask=attn,  # no bidirectionality in the image block
    )
    diff = np.abs(
        np.asarray(out_bidir.logits, dtype=np.float32)
        - np.asarray(out_causal.logits, dtype=np.float32)
    ).max()
    self.assertGreater(
        float(diff),
        1e-4,
        'logits must change when the sliding mask loses the bidirectional'
        ' image block — otherwise the external mask is not consumed',
    )

  def test_sliding_mask_sized_to_cache_is_accepted(self):
    """Masks of shape [B, L, C] with C > L work (inference prefill geometry
    — the stock internal [B, L, L] builder would be shape-wrong there)."""
    cache_length = P + 8
    cache = self.model.config.init_cache(
        batch_size=2, dtype=jnp.bfloat16, cache_length=cache_length
    )
    token_mask = self.tokens != 0
    attn, sliding = vision_mask_helpers.make_vision_prefill_masks(
        tokens=self.tokens, token_mask=token_mask, cache_length=cache_length
    )
    positions = jnp.cumsum(token_mask, axis=-1) - (
        jnp.cumsum(token_mask, axis=-1) >= 1
    )
    out = self.model.apply(
        self.variables,
        self.tokens,
        images=(self.patches, self.positions_xy),
        cache=cache,
        positions=positions,
        attention_mask=attn,
        sliding_attention_mask=sliding,
    )
    self.assertEqual(out.logits.shape, (2, P, VOCAB))
    layer0 = out.cache['layer_0']
    self.assertEqual(layer0['k'].shape[1], cache_length)
    np.testing.assert_array_equal(np.asarray(layer0['end_index']), [P, P])

  ##############################################################################
  # Fail-fast against the stock input convention.
  ##############################################################################

  def test_stock_model_rejects_batched_tuple_images(self):
    """The stock Transformer typecheck rejects the batched tuple — the two
    input conventions cannot be silently mixed."""
    stock = gemma_diffusion.DiffusionGemma_26B_A4B(
        config=vision_test_utils.make_standalone_config(), text_only=False
    )
    with self.assertRaises(
        (ktyping_errors.KTypeCheckError, TypeError, AttributeError)
    ):
      stock.apply(
          self.variables,
          self.tokens,
          images=(self.patches, self.positions_xy),
      )


if __name__ == '__main__':
  absltest.main()
