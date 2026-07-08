"""Canonical-usage unit test for ``vision_sft_model.sft_encode``.

One toy example with the parameter-free outputs written out verbosely.
(The encoder logits and cache K/V values depend on network parameters;
their numeric behavior is covered by the behavior/property tests in
``vision_input_test.py`` and ``vision_train_infer_consistency_test.py``.)
"""

from typing import Any

from absl.testing import absltest
import flax.linen as nn
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_sft_model
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_test_utils
import jax
import jax.numpy as jnp
import numpy as np

VOCAB = vision_test_utils.STANDALONE_VOCAB


class _EncodeWrapper(nn.Module):
  gemma_network: Any

  @nn.compact
  def __call__(self, prompt, x0_tokens, canvas_mask, patches, positions_xy):
    return vision_sft_model.sft_encode(
        gemma_network=self.gemma_network,
        prompt=prompt,
        x0_tokens=x0_tokens,
        canvas_mask=canvas_mask,
        selected_canvas_idx=jnp.zeros((prompt.shape[0],), jnp.int32),
        prompt_len=6,
        total_canvas_len=4,
        canvas_size=4,
        images=(patches, positions_xy),
    )


class SftEncodeCanonicalUsageTest(absltest.TestCase):

  def test_sft_encode_canonical_usage(self):
    """P=6, one canvas of 4 tokens, one image with S_v=2.

    Inputs:

        prompt (position 0..5):    2    5    -2    -2     6     0
                                  bos  text  soft  soft  text  PAD
        x0 canvas (position 6..9): 7    8     9     1(eos)   all valid
        selected_canvas_idx:       0

    The full encoder sequence is prompt ++ x0 (10 positions).
    """
    from ft_gemma.diffusion.hackable_diffusion_adapter.hd import (  # pylint: disable=g-import-not-at-top
        vision_hd_gemma_network,
    )

    gemma_network = (
        vision_hd_gemma_network.VisionWrappedDiffusionGemmaNetwork(
            gemma_model=vision_test_utils.make_standalone_vision_model()
        )
    )
    wrapper = _EncodeWrapper(gemma_network=gemma_network)

    prompt = jnp.asarray([[2, 5, -2, -2, 6, 0]], dtype=jnp.int32)
    x0_tokens = jnp.asarray([[7, 8, 9, 1]], dtype=jnp.int32)
    canvas_mask = jnp.asarray([[True, True, True, True]])
    patches, positions_xy, s_v = vision_test_utils.make_grid_image(6, 3, 0)
    self.assertEqual(s_v, 2)
    patches = jnp.asarray(patches)[None]
    positions_xy = jnp.asarray(positions_xy)[None]

    args = (prompt, x0_tokens, canvas_mask, patches, positions_xy)
    variables = wrapper.init(jax.random.PRNGKey(0), *args)
    encoder_logits, kv_cache, positions, prompt_mask = wrapper.apply(
        variables, *args
    )

    # prompt_mask: PAD is invalid; the -2 soft-token slots are NON-PAD, so
    # they count as real prompt tokens with no code change (design rule).
    np.testing.assert_array_equal(
        prompt_mask,
        #  bos  text  soft  soft  text  PAD
        [[1, 1, 1, 1, 1, 0]],
    )

    # RoPE positions over prompt ++ x0: cumsum(valid) - 1. The PAD at
    # position 5 repeats the previous value; the canvas continues from it.
    np.testing.assert_array_equal(
        positions,
        # pos:   0  1  2  3  4  5(P) 6  7  8  9
        [[0, 1, 2, 3, 4, 4, 5, 6, 7, 8]],
    )

    # Cache write cursor rewound to prompt_len + selected_idx * canvas_size
    # = 6 + 0 * 4 = 6, in every layer.
    for layer_name in ('layer_0', 'layer_1', 'layer_2'):
      np.testing.assert_array_equal(
          np.asarray(kv_cache[layer_name]['end_index']), [6]
      )

    # Full-length AR logits: one row per position of prompt ++ x0
    # (remove_mm_logits bypassed).
    self.assertEqual(encoder_logits.shape, (1, 10, VOCAB))


if __name__ == '__main__':
  absltest.main()
