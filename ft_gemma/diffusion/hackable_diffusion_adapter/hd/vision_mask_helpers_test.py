"""Unit tests for ``make_vision_prefill_masks``.

Intended behavior (design §5.1, encoder-call diagram):
  * The first returned mask is byte-identical to the baseline causal prefill
    mask — ``mask[b, q, k] = (k <= q) AND valid[b, k]``, right-padded to the
    cache length. It is what the GLOBAL layers consume: image tokens stay
    strictly causal there.
  * The second (sliding) mask differs from the first ONLY inside each
    contiguous image soft-token block, whose ``-2`` slots see each other
    fully. It is what the LOCAL_SLIDING layers consume.
  * Two image blocks separated by text do NOT attend to each other.
  * Both masks are sized to the cache axis so the same builder works at
    inference prefill where cache_length > seq_len.
"""

from absl.testing import absltest
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_mask_helpers
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
import jax.numpy as jnp
import numpy as np


def _masks(tokens, cache_length):
  tokens = jnp.asarray(tokens, dtype=jnp.int32)
  token_mask = tokens != 0
  attn, sliding = vision_mask_helpers.make_vision_prefill_masks(
      tokens=tokens, token_mask=token_mask, cache_length=cache_length
  )
  return np.asarray(attn), np.asarray(sliding), np.asarray(token_mask)


class MakeVisionPrefillMasksTest(absltest.TestCase):

  # Layout: t0 t1 <soi> -2 -2 -2 <eoi> t2 PAD PAD   (one image, S_v=3)
  TOKENS = [[4, 5, 9, -2, -2, -2, 10, 6, 0, 0]]
  SOFT = slice(3, 6)  # the -2 slots

  def test_canonical_usage(self):
    """One toy example, inputs and BOTH output masks fully written out.

    Input: one row of 6 tokens — text, <soi>-like marker, two -2 soft-token
    slots, <eoi>-like marker, PAD — with an 8-slot cache:

        position:  0    1      2     3     4      5
        token:     4    9     -2    -2    10      0
                  text  <soi> soft  soft  <eoi>  PAD
    """
    tokens = jnp.asarray([[4, 9, -2, -2, 10, 0]], dtype=jnp.int32)
    token_mask = tokens != 0
    attn, sliding = vision_mask_helpers.make_vision_prefill_masks(
        tokens=tokens, token_mask=token_mask, cache_length=8
    )

    # Causal mask (GLOBAL layers): key k visible iff k <= q AND k is not
    # PAD. Columns 6-7 are the not-yet-written cache slots.
    expected_attn = [
        # keys:  4  9 -2 -2 10  P  .  .
        [1, 0, 0, 0, 0, 0, 0, 0],  # q0: text        sees itself only
        [1, 1, 0, 0, 0, 0, 0, 0],  # q1: <soi>
        [1, 1, 1, 0, 0, 0, 0, 0],  # q2: soft token  strictly causal here
        [1, 1, 1, 1, 0, 0, 0, 0],  # q3: soft token
        [1, 1, 1, 1, 1, 0, 0, 0],  # q4: <eoi>
        [1, 1, 1, 1, 1, 0, 0, 0],  # q5: PAD row (PAD key col 5 stays hidden)
    ]
    np.testing.assert_array_equal(np.asarray(attn[0], dtype=int), expected_attn)

    # Sliding mask (LOCAL_SLIDING layers): identical, except the two soft
    # tokens (positions 2 and 3) see each other bidirectionally — the single
    # added entry is q2 -> k3.
    expected_sliding = [
        # keys:  4  9 -2 -2 10  P  .  .
        [1, 0, 0, 0, 0, 0, 0, 0],  # q0: text
        [1, 1, 0, 0, 0, 0, 0, 0],  # q1: <soi>
        [1, 1, 1, 1, 0, 0, 0, 0],  # q2: soft token  <- sees soft k3 (ahead)
        [1, 1, 1, 1, 0, 0, 0, 0],  # q3: soft token
        [1, 1, 1, 1, 1, 0, 0, 0],  # q4: <eoi>       causal again
        [1, 1, 1, 1, 1, 0, 0, 0],  # q5: PAD row
    ]
    np.testing.assert_array_equal(
        np.asarray(sliding[0], dtype=int), expected_sliding
    )

  def test_causal_mask_matches_baseline_helper(self):
    """First output == baseline make_causal_prefill_mask (unchanged rule)."""
    attn, _, token_mask = _masks(self.TOKENS, cache_length=10)
    baseline = np.asarray(
        mask_helpers.make_causal_prefill_mask(
            jnp.asarray(token_mask), cache_length=10
        )
    )
    np.testing.assert_array_equal(attn, baseline)

  def test_causal_mask_is_tril_and_valid(self):
    """mask[q, k] = (k <= q) AND valid[k] — image tokens causal (GLOBAL)."""
    attn, _, token_mask = _masks(self.TOKENS, cache_length=10)
    expected = np.tril(np.ones((10, 10), bool)) & token_mask[0][None, :]
    np.testing.assert_array_equal(attn[0], expected)
    # A soft-token query must NOT see a later soft token in this mask.
    self.assertFalse(attn[0, 3, 5])

  def test_sliding_mask_bidirectional_exactly_within_image_block(self):
    """sliding == causal OR (q and k both in the same image block)."""
    attn, sliding, _ = _masks(self.TOKENS, cache_length=10)
    img = np.zeros(10, bool)
    img[self.SOFT] = True
    expected = attn[0] | (img[:, None] & img[None, :])
    np.testing.assert_array_equal(sliding[0], expected)
    # Explicit probes of the design's ASCII example:
    self.assertTrue(sliding[0, 3, 5])  # soft q sees LATER soft k (lookahead)
    self.assertTrue(sliding[0, 5, 3])  # ...and earlier soft k
    self.assertFalse(sliding[0, 3, 6])  # but not past the block (<eoi>)
    self.assertFalse(sliding[0, 2, 3])  # text before the block: still causal
    self.assertFalse(sliding[0, 1, 3])  # earlier text never sees soft tokens

  def test_two_image_blocks_do_not_attend_each_other(self):
    """Bidirectionality is per CONTIGUOUS block (multi-image isolation)."""
    tokens = [[4, -2, -2, 5, -2, -2, 6, 0]]
    attn, sliding, _ = _masks(tokens, cache_length=8)
    # Within-block lookahead works...
    self.assertTrue(sliding[0, 1, 2])
    self.assertTrue(sliding[0, 4, 5])
    # ...but block 1 gets NO lookahead into block 2.
    self.assertFalse(sliding[0, 1, 4])
    self.assertFalse(sliding[0, 2, 5])
    # Backward visibility across blocks exists — but only via the plain
    # causal rule, identically to the causal mask (no extra visibility).
    self.assertTrue(attn[0, 4, 1])
    block_1, block_2 = np.zeros(8, bool), np.zeros(8, bool)
    block_1[[1, 2]] = True
    block_2[[4, 5]] = True
    cross_block = (block_1[:, None] & block_2[None, :]) | (
        block_2[:, None] & block_1[None, :]
    )
    # Wherever q and k are soft tokens of DIFFERENT blocks, sliding == causal.
    np.testing.assert_array_equal(
        sliding[0][cross_block], attn[0][cross_block]
    )

  def test_pad_keys_hidden_from_all_queries(self):
    attn, sliding, token_mask = _masks(self.TOKENS, cache_length=10)
    pad_cols = ~token_mask[0]
    self.assertFalse(attn[0][:, pad_cols].any())
    self.assertFalse(sliding[0][:, pad_cols].any())

  def test_masks_padded_to_cache_length(self):
    """cache_length > seq_len (inference prefill): extra key columns False."""
    seq_len, cache_length = len(self.TOKENS[0]), 16
    attn, sliding, _ = _masks(self.TOKENS, cache_length=cache_length)
    self.assertEqual(attn.shape, (1, seq_len, cache_length))
    self.assertEqual(sliding.shape, (1, seq_len, cache_length))
    self.assertFalse(attn[0][:, seq_len:].any())
    self.assertFalse(sliding[0][:, seq_len:].any())
    # The first seq_len columns are identical to the unpadded build.
    attn_sq, sliding_sq, _ = _masks(self.TOKENS, cache_length=seq_len)
    np.testing.assert_array_equal(attn[0][:, :seq_len], attn_sq[0])
    np.testing.assert_array_equal(sliding[0][:, :seq_len], sliding_sq[0])

  def test_no_image_degenerates_to_causal(self):
    tokens = [[4, 5, 6, 0]]
    attn, sliding, _ = _masks(tokens, cache_length=4)
    np.testing.assert_array_equal(attn, sliding)


if __name__ == "__main__":
  absltest.main()
