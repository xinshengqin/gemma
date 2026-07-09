"""Vision-input consumption and padding tests for the vision SFT stack.

Uses one fake visual-sudoku example and the tiny model config
(``sft_sudoku_vision_full_tiny.py``) to run the full
``VisionSFTDiffusion.__call__`` training forward (fixed rngs) and check:

1. **Vision input is consumed** — perturbing the image (the real, non-padding
   patches) changes BOTH training losses (encoder AR loss and diffusion
   loss): the image reaches the AR logits directly and the denoiser through
   the prompt K/V in the cache.

2. **Vision input is padded correctly** — perturbing "padding" positions
   leaves both losses bit-identical:
     a. padding *patches* (positions_xy == -1) never contribute — they are
        masked out of the vision-tower attention and pool with weight 0;
     b. padding *soft tokens* (tower output rows S_v..S_max-1, mask False)
        are discarded by ``merge_flat_embeddings`` (excess rows scatter to
        slot 0, which is then restored).
"""

import contextlib

from absl.testing import absltest
import flax.linen as nn
from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_test_utils
from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model
from gemma.gm.nn.gemma4.vision import _encoder as gemma4_vision
import jax
import jax.numpy as jnp
import numpy as np
import optax


def _perturb_padding_soft_tokens_interceptor(next_fun, args, kwargs, context):
  """Adds a large offset to the vision tower's PADDING soft-token rows."""
  outputs = next_fun(*args, **kwargs)
  if (
      isinstance(context.module, gemma4_vision.VisionEncoder)
      and context.method_name == '__call__'
  ):
    perturbed = []
    for embeddings, mask in outputs:
      assert mask is not None
      embeddings = jnp.where(mask[..., None], embeddings, embeddings + 123.0)
      perturbed.append((embeddings, mask))
    return tuple(perturbed)
  return outputs


class VisionInputTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.model = vision_test_utils.make_tiny_model()
    cls.batch = vision_test_utils.make_fake_batch(seed=0, batch_size=1)
    rng = jax.random.PRNGKey(42)
    cls.variables = cls.model.init(
        {'params': rng, 'sampling': rng},
        **vision_test_utils.model_kwargs_from_batch(cls.batch),
        is_training=False,
    )
    cls.base_losses = cls._losses()

  @classmethod
  def _losses(cls, patches=None, interceptor=None):
    """Runs the full training forward (fixed rngs) and returns both losses.

    Args:
      patches: Optional override for ``batch.patches``.
      interceptor: Optional Flax method interceptor active during the
        forward pass.

    Returns:
      (encoder_loss, diffusion_loss) as Python floats — per-example values
      averaged over the batch, computed exactly like the config's
      ``EncoderARLoss`` and ``NoWeightDiscreteLoss(mask=target_mask)``.
    """
    kwargs = vision_test_utils.model_kwargs_from_batch(cls.batch)
    if patches is not None:
      kwargs['patches'] = patches

    context = (
        nn.intercept_methods(interceptor)
        if interceptor is not None
        else contextlib.nullcontext()
    )
    with context:
      preds = cls.model.apply(
          cls.variables,
          **kwargs,
          is_training=False,
          rngs={'sampling': jax.random.PRNGKey(7)},
      )

    # Encoder AR loss (formula of sft_model.EncoderARLoss).
    encoder_loss = sft_model.EncoderARLoss().get_values(
        encoder_logits=preds['encoder_logits'].astype(jnp.float32),
        encoder_target=preds['encoder_target'],
        encoder_target_mask=preds['encoder_target_mask'].astype(jnp.float32),
    )

    # Diffusion loss: masked CE(pass-2 logits, clean x0) over target_mask.
    logits = preds['output']['logits'].astype(jnp.float32)  # [B, TC, V]
    x0 = jnp.asarray(cls.batch['canvas'])[..., 0]  # [B, TC]
    mask = preds['target']['target_mask'][..., 0].astype(jnp.float32)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, x0)
    diffusion_loss = jnp.sum(ce * mask, axis=-1) / jnp.maximum(
        jnp.sum(mask, axis=-1), 1.0
    )

    return float(jnp.mean(encoder_loss)), float(jnp.mean(diffusion_loss))

  def test_perturbing_image_changes_losses(self):
    """Vision input is consumed: a different image -> different losses."""
    patches = jnp.asarray(self.batch['patches'])
    positions_xy = np.asarray(self.batch['positions_xy'])
    is_real_patch = (positions_xy != -1).any(axis=-1)  # [B, P_p]
    perturbed = jnp.where(
        jnp.asarray(is_real_patch)[..., None], patches + 0.5, patches
    )

    encoder_loss, diffusion_loss = self._losses(patches=perturbed)
    base_encoder_loss, base_diffusion_loss = self.base_losses

    self.assertNotAlmostEqual(
        encoder_loss,
        base_encoder_loss,
        places=6,
        msg='encoder AR loss must change when the image content changes',
    )
    self.assertNotAlmostEqual(
        diffusion_loss,
        base_diffusion_loss,
        places=6,
        msg=(
            'diffusion loss must change when the image content changes (the'
            ' denoiser sees the image through the prompt K/V in the cache)'
        ),
    )

  def test_perturbing_padding_patches_keeps_losses(self):
    """Padding patches (positions_xy == -1) must not affect the losses."""
    patches = jnp.asarray(self.batch['patches'])
    positions_xy = np.asarray(self.batch['positions_xy'])
    is_real_patch = (positions_xy != -1).any(axis=-1)  # [B, P_p]
    self.assertGreater(int((~is_real_patch).sum()), 0)
    perturbed = jnp.where(
        jnp.asarray(is_real_patch)[..., None], patches, patches + 123.0
    )

    encoder_loss, diffusion_loss = self._losses(patches=perturbed)
    base_encoder_loss, base_diffusion_loss = self.base_losses

    self.assertEqual(
        encoder_loss,
        base_encoder_loss,
        msg='encoder AR loss changed when only PADDING patches were perturbed',
    )
    self.assertEqual(
        diffusion_loss,
        base_diffusion_loss,
        msg='diffusion loss changed when only PADDING patches were perturbed',
    )

  def test_perturbing_padding_soft_tokens_keeps_losses(self):
    """Padding soft tokens (rows S_v..S_max-1) must not affect the losses."""
    encoder_loss, diffusion_loss = self._losses(
        interceptor=_perturb_padding_soft_tokens_interceptor
    )
    base_encoder_loss, base_diffusion_loss = self.base_losses

    self.assertEqual(
        encoder_loss,
        base_encoder_loss,
        msg=(
            'encoder AR loss changed when only PADDING soft tokens were'
            ' perturbed — merge_flat_embeddings must discard them'
        ),
    )
    self.assertEqual(
        diffusion_loss,
        base_diffusion_loss,
        msg=(
            'diffusion loss changed when only PADDING soft tokens were'
            ' perturbed — merge_flat_embeddings must discard them'
        ),
    )


if __name__ == '__main__':
  absltest.main()
