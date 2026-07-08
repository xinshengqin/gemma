"""Train/inference consistency test for the vision SFT stack.

Uses one fake visual-sudoku example and the tiny model config
(``sft_sudoku_vision_full_tiny.py``) to run one *training-mode* forward pass
(``vision_sft_model.sft_encode`` + ``sft_decode`` — the exact functions
``VisionSFTDiffusion.__call__`` uses) and one *inference-mode* pass
(``VisionGemmaARStateHandler.init_ar_state`` + ``SFTInferenceFn`` — the
exact path the sampling evaluator uses), with the same parameters, and
asserts the intermediate results are identical:

  * the decoder attention masks,
  * the prompt-region KV cache written by the encoder prefill (activations —
    this includes the merged image soft tokens, since the prompt K/V are a
    function of them),
  * the RoPE positions and cache write cursors,
  * the encoder prefill logits over the real prompt tokens,
  * the denoiser logits (train pass 1 vs the first inference denoising step,
    both with zero self-conditioning and the same x_t).
"""

from typing import Any

from absl.testing import absltest
import flax.linen as nn
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import (
    vision_ar_state_handler,
)
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import (
    vision_hd_gemma_network,
)
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_sft_model
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_test_utils
from gemma import gm
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
import jax
import jax.numpy as jnp
import numpy as np

P = vision_test_utils.PROMPT_LEN
CS = vision_test_utils.CANVAS_SIZE
NC = vision_test_utils.NUM_CANVASES
TC = vision_test_utils.TOTAL_CANVAS_LEN
FS = vision_test_utils.FULL_SEQ_LEN


class _TrainSideWrapper(nn.Module):
  """Runs the training-mode encoder prefill + denoiser pass 1."""

  gemma_network: Any

  @nn.compact
  def __call__(self, prompt, x0_tokens, canvas_mask, patches, positions_xy, xt):
    batch_size = prompt.shape[0]
    selected_canvas_idx = jnp.zeros((batch_size,), dtype=jnp.int32)
    time = jnp.ones((batch_size, TC, 1), dtype=jnp.float32)

    encoder_logits, kv_cache, positions, prompt_mask = (
        vision_sft_model.sft_encode(
            gemma_network=self.gemma_network,
            prompt=prompt,
            x0_tokens=x0_tokens,
            canvas_mask=canvas_mask,
            selected_canvas_idx=selected_canvas_idx,
            prompt_len=P,
            total_canvas_len=TC,
            canvas_size=CS,
            images=(patches, positions_xy),
        )
    )
    denoiser_output = vision_sft_model.sft_decode(
        gemma_network=self.gemma_network,
        xt=xt,
        time=time,
        kv_cache=kv_cache,
        positions=positions,
        prompt_mask=prompt_mask,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        prompt_len=P,
        total_canvas_len=TC,
        canvas_size=CS,
        is_training=False,
    )
    decoder_mask = mask_helpers.create_decoder_attention_mask(
        prompt_mask=prompt_mask,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        prompt_len=P,
        total_canvas_len=TC,
        canvas_size=CS,
        num_queries=TC,
    )
    return {
        'encoder_logits': encoder_logits,
        'kv_cache': kv_cache,
        'positions': positions,
        'decoder_mask': decoder_mask,
        'logits': denoiser_output['logits'],
    }


class _PromptPrefillWrapper(nn.Module):
  """Runs the inference-mode prompt-only prefill (to read its logits)."""

  gemma_network: Any

  @nn.compact
  def __call__(self, prompt, prompt_mask, patches, positions_xy):
    kv_cache, logits, positions, _ = (
        vision_hd_gemma_network.prefill_kv_cache_with_encoder(
            tokens=prompt,
            input_mask=prompt_mask,
            init_cache_fn=self.gemma_network.init_cache,
            encoder_fn=self.gemma_network.encoder_call,
            cache_length=P + NC * CS,
            images=(patches, positions_xy),
        )
    )
    return {'kv_cache': kv_cache, 'logits': logits, 'positions': positions}


class VisionTrainInferConsistencyTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.rng = jax.random.PRNGKey(42)
    cls.model = vision_test_utils.make_tiny_model()
    cls.batch = vision_test_utils.make_fake_batch(seed=0, batch_size=1)
    cls.gemma_network = cls.model.gemma_network

    batch = cls.batch
    cls.prompt = jnp.asarray(batch['prompt'])
    cls.x0_tokens = jnp.asarray(batch['canvas'])[..., 0]
    cls.canvas_mask = jnp.asarray(batch['canvas_mask'])
    cls.patches = jnp.asarray(batch['patches'])
    cls.positions_xy = jnp.asarray(batch['positions_xy'])

    # Same noisy canvas for both regimes (contents are irrelevant — only
    # equality across the two paths matters). The network ignores `time`.
    vocab = cls.gemma_network.num_embed
    cls.xt = jax.random.randint(
        jax.random.PRNGKey(0), (1, TC, 1), minval=0, maxval=vocab
    )

    # ---------------- training-mode forward pass ----------------
    wrapper = _TrainSideWrapper(gemma_network=cls.gemma_network)
    wrapper_args = (
        cls.prompt,
        cls.x0_tokens,
        cls.canvas_mask,
        cls.patches,
        cls.positions_xy,
        cls.xt,
    )
    cls.variables = wrapper.init(cls.rng, *wrapper_args)
    cls.train_out = wrapper.apply(cls.variables, *wrapper_args)

    # ---------------- inference-mode pass ----------------
    gemma_network_params = cls.variables['params']['gemma_network']
    handler = vision_ar_state_handler.VisionGemmaARStateHandler(
        gemma_network=cls.gemma_network,
        gemma_params=gemma_network_params['gemma_model'],
        pad_token=gm.text.Gemma4Tokenizer.special_tokens.PAD,
        end_tokens=(
            gm.text.Gemma4Tokenizer.special_tokens.EOS,
            gm.text.Gemma4Tokenizer.special_tokens.END_OF_TURN,
            gm.text.Gemma4Tokenizer.special_tokens.BEGIN_OF_TOOL_RESPONSE,
        ),
    )
    prompt_lengths = jnp.sum(cls.prompt != 0, axis=-1)
    state = handler.init_ar_state(
        batch_size=1,
        conditioning={
            'prompt_tokens': cls.prompt,
            'prompt_lengths': prompt_lengths,
            'patches': cls.patches,
            'positions_xy': cls.positions_xy,
        },
        canvas_length=CS,
        max_num_canvases=NC,
    )
    cls.infer_cond = handler.create_conditioning_from_state(state)

    inference_fn = vision_sft_model.SFTInferenceFn(
        gemma_network=cls.gemma_network,
        params=gemma_network_params,
    )
    # First denoising step: no sc_logits in the conditioning -> the network
    # substitutes zeros, exactly like training pass 1.
    time = jnp.ones((1, TC, 1), dtype=jnp.float32)
    cls.infer_out = inference_fn(
        time=time, xt=cls.xt, conditioning=dict(cls.infer_cond)
    )

    # Inference prompt prefill logits (same call init_ar_state makes, with
    # the logits kept).
    prefill = _PromptPrefillWrapper(gemma_network=cls.gemma_network)
    prompt_mask = cls.prompt != 0
    cls.prefill_out = prefill.apply(
        cls.variables, cls.prompt, prompt_mask, cls.patches, cls.positions_xy
    )

  def test_attention_masks_identical(self):
    """Train decoder mask [B,TC,FS] == inference canvas mask [B,CS,C]."""
    np.testing.assert_array_equal(
        np.asarray(self.train_out['decoder_mask']),
        np.asarray(self.infer_cond['attention_mask']),
        err_msg='decoder attention masks differ between train and inference',
    )

  def test_positions_identical(self):
    """Canvas RoPE positions match between the two regimes."""
    np.testing.assert_array_equal(
        np.asarray(self.train_out['positions'][:, P:]),
        np.asarray(self.infer_cond['positions']),
        err_msg='canvas positions differ between train and inference',
    )

  def test_prompt_kv_cache_identical(self):
    """Prompt-region K/V activations (incl. merged image K/V) match."""
    train_cache = self.train_out['kv_cache']
    infer_cache = self.infer_cond['kv_cache']
    self.assertEqual(set(train_cache.keys()), set(infer_cache.keys()))
    for layer_name in train_cache:
      # Both regimes leave the write cursor at the canvas start (= P).
      np.testing.assert_array_equal(
          np.asarray(train_cache[layer_name]['end_index']),
          np.asarray(infer_cache[layer_name]['end_index']),
      )
      for key in ('k', 'v'):
        np.testing.assert_array_equal(
            np.asarray(train_cache[layer_name][key][:, :P]),
            np.asarray(infer_cache[layer_name][key][:, :P]),
            err_msg=(
                f'prompt-region KV cache {layer_name}/{key} differs between'
                ' train and inference'
            ),
        )

  def test_encoder_prefill_logits_identical(self):
    """Encoder logits over the real prompt tokens match the prefill's."""
    real_len = int(np.asarray(self.prompt != 0).sum())
    np.testing.assert_array_equal(
        np.asarray(self.train_out['encoder_logits'][:, :real_len]),
        np.asarray(self.prefill_out['logits'][:, :real_len]),
        err_msg=(
            'encoder logits at real prompt positions differ between the'
            ' training prefill and the inference prompt prefill'
        ),
    )

  def test_denoiser_logits_identical(self):
    """Train pass-1 logits == first inference denoising step logits."""
    np.testing.assert_array_equal(
        np.asarray(self.train_out['logits']),
        np.asarray(self.infer_out['logits']),
        err_msg='denoiser logits differ between train and inference',
    )


if __name__ == '__main__':
  absltest.main()
