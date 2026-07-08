"""Canonical-usage unit test for ``VisionGemmaARStateHandler.init_ar_state``.

One toy example with every parameter-free field of the initial sampler
state written out verbosely. (The prefilled cache K/V values depend on
network parameters; that the image K/V actually lands in the cache is
covered by ``vision_hd_gemma_network_test.py`` and the train/inference
consistency test.)
"""

from absl.testing import absltest
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import (
    vision_ar_state_handler,
)
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import (
    vision_hd_gemma_network,
)
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_mask_helpers
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_test_utils
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
import jax
import jax.numpy as jnp
import numpy as np


class InitArStateCanonicalUsageTest(absltest.TestCase):

  def test_init_ar_state_canonical_usage(self):
    """P=6 prompt with a 2-slot image span; 1 canvas of 4 tokens (C=10).

    Conditioning:

        prompt_tokens (position 0..5):  2    5    -2    -2     6     0
                                       bos  text  soft  soft  text  PAD
        prompt_lengths:                 [5]  (the -2 slots count: non-PAD)
        patches / positions_xy:         one 6x3-patch image (S_v = 2)
    """
    net = vision_hd_gemma_network.VisionWrappedDiffusionGemmaNetwork(
        gemma_model=vision_test_utils.make_standalone_vision_model()
    )
    prompt_tokens = jnp.asarray([[2, 5, -2, -2, 6, 0]], dtype=jnp.int32)
    input_mask = prompt_tokens != 0
    patches, positions_xy, s_v = vision_test_utils.make_grid_image(6, 3, 0)
    self.assertEqual(s_v, 2)
    patches = jnp.asarray(patches)[None]
    positions_xy = jnp.asarray(positions_xy)[None]

    # Init the network params through the images-enabled encoder path.
    cache = net.gemma_model.config.init_cache(
        batch_size=1, dtype=jnp.bfloat16, cache_length=10
    )
    attn, sliding = vision_mask_helpers.make_vision_prefill_masks(
        tokens=prompt_tokens, token_mask=input_mask, cache_length=10
    )
    variables = net.init(
        jax.random.PRNGKey(0),
        x=prompt_tokens,
        conditioning_embeddings={
            'kv_cache': cache,
            'positions': mask_helpers.build_positions_from_mask(input_mask),
            'attention_mask': attn,
            'sliding_attention_mask': sliding,
            'images': (patches, positions_xy),
        },
        method=net.encoder_call,
    )

    handler = vision_ar_state_handler.VisionGemmaARStateHandler(
        gemma_network=net,
        gemma_params=variables['params']['gemma_model'],
        end_tokens=(1,),
    )
    state = handler.init_ar_state(
        batch_size=1,
        conditioning={
            'prompt_tokens': prompt_tokens,
            'prompt_lengths': jnp.asarray([5]),
            'patches': patches,
            'positions_xy': positions_xy,
        },
        canvas_length=4,
        max_num_canvases=1,
    )

    # Cache capacity C = P + NC * CS = 6 + 1 * 4 = 10; the prompt prefill
    # (including the merged image) left the write cursor at P = 6.
    self.assertEqual(state['kv_cache']['layer_0']['k'].shape, (1, 10, 2, 8))
    np.testing.assert_array_equal(
        np.asarray(state['kv_cache']['layer_0']['end_index']), [6]
    )

    # Which cache slots hold REAL (written) K/V after the prompt prefill:
    #
    #   slot:     0    1    2     3     4     5     6  7  8  9
    #   content: bos  text soft  soft  text  PAD    -  -  -  -
    #   written:  y    y    y     y     y     y     n  n  n  n
    #
    # All six prompt slots are written — slots 2-3 hold the image-derived
    # K/V, and the PAD slot 5 holds garbage that stays invisible because
    # full_attention_mask (below) hides key column 5 permanently. The four
    # canvas slots are still empty; the sampler fills them per canvas.
    k = np.asarray(state['kv_cache']['layer_0']['k'], dtype=np.float32)
    for slot in (0, 1, 2, 3, 4, 5):
      self.assertTrue(np.any(k[0, slot] != 0.0), f'slot {slot} not written')
    np.testing.assert_array_equal(k[0, 6:], 0.0)

    # The output buffer starts as the prompt followed by empty canvas slots,
    # and the write position for the first canvas is P = 6.
    np.testing.assert_array_equal(
        state['predicted_tokens'],
        #  --------- prompt ---------    ---- canvas 0 ----
        [[2, 5, -2, -2, 6, 0, 0, 0, 0, 0]],
    )
    self.assertEqual(state['step'], 6)
    np.testing.assert_array_equal(state['done'], [False])

    # Canvas RoPE positions continue after the LAST REAL prompt token
    # (prompt_lengths = 5, so the first canvas token sits at position 5).
    np.testing.assert_array_equal(state['positions'], [[5, 6, 7, 8]])

    # Valid prompt tokens; the -2 soft slots are marked real.
    np.testing.assert_array_equal(
        state['prompt_mask'],
        [[1, 1, 1, 1, 1, 0]],
    )

    # Permanent pad mask over the cache: prompt validity, then all-True
    # future decode slots.
    np.testing.assert_array_equal(
        state['full_attention_mask'],
        #  --------- prompt ---------  -- canvas 0 --
        [[1, 1, 1, 1, 1, 0, 1, 1, 1, 1]],
    )

    # Decoder mask for the first canvas: every canvas query sees the valid
    # prompt keys — the image span included — plus all of canvas 0; the
    # prompt PAD slot stays hidden.
    np.testing.assert_array_equal(
        np.asarray(state['attention_mask'][0], dtype=int),
        [
            # keys:  2  5 -2 -2  6  P  c0 c1 c2 c3
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # canvas query 0
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # canvas query 1
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # canvas query 2
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # canvas query 3
        ],
    )


if __name__ == '__main__':
  absltest.main()
