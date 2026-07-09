"""Vision-input SFT model for DiffusionGemma (add-visual-inputs design).

Overrides the pieces of the baseline SFT stack the design marks [DESIGN]
(docs/add-visual-inputs/index.html §3/§5/§6):

* ``sft_encode`` — the encoder prefill now also receives
  ``images=(patches, positions_xy)`` and forwards them into
  ``Transformer.__call__`` where the vision tower runs and the soft tokens
  are merged into the prompt embeddings — so the prefilled KV cache contains
  image K/V. The vision-aware prefill masks are built inside
  ``vision_hd_gemma_network.prefill_kv_cache_with_encoder``.

* ``VisionSFTDiffusion`` — adds the ``patches`` / ``positions_xy`` kontext
  keys (bound to ``batch.patches`` / ``batch.positions_xy``) and threads the
  pair into ``sft_encode``. Time sampling, corruption, canvas selection,
  self-conditioning, both denoiser passes and both loss formulas are the
  baseline ones — the denoiser never sees pixels or patches; the image
  reaches it only as cached K/V.

* ``VisionGemmaSamplingEvaluator`` — the eval entry point adds the patch
  tensors to the conditioning dict, so the prompt prefill inside
  ``init_ar_state`` embeds and merges them (design §6, §11 step 1).
"""

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from kauldron import kd

from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_hd_gemma_network
from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers
from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model

PAD_TOKEN = sft_model.PAD_TOKEN

# Re-exported so configs/tests can reference the unchanged pieces from here.
SFTInferenceFn = sft_model.SFTInferenceFn
EncoderARLoss = sft_model.EncoderARLoss
GemmaKDARSampler = sft_model.GemmaKDARSampler
sft_decode = sft_model.sft_decode


def sft_encode(
    gemma_network: Any,
    *,
    prompt: jnp.ndarray,
    x0_tokens: jnp.ndarray,
    canvas_mask: jnp.ndarray,
    selected_canvas_idx: jnp.ndarray,
    prompt_len: int,
    total_canvas_len: int,
    canvas_size: int,
    images: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    pad_token: int = PAD_TOKEN,
) -> tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray]:
  """Runs the SFT encoder pass with vision inputs.

  Same contract as ``sft_model.sft_encode`` plus ``images``: one standard
  causal Gemma forward over ``prompt ++ x0 [B, FS]``, with the vision tower
  running inside and the soft tokens merged at the ``-2`` slots of the
  prompt. The prompt validity rule is unchanged — the ``-2`` soft-token
  slots are non-PAD, so ``prompt != pad_token`` marks them valid with no
  code change.

  Args:
    gemma_network: The Gemma backbone wrapper (bound or unbound).
    prompt: Prompt tokens with the image span pre-expanded, ``[B, P]``.
    x0_tokens: Target response tokens, ``[B, TotalCanvasLen]``.
    canvas_mask: Valid-canvas mask, shape ``[B, TotalCanvasLen]``.
    selected_canvas_idx: Per-example selected canvas index, shape ``[B]``.
    prompt_len: Fixed padded maximum prompt length.
    total_canvas_len: Total canvas length.
    canvas_size: Number of tokens per canvas.
    images: Optional ``(patches [B, P_p, p_d], positions_xy [B, P_p, 2])``.
    pad_token: Padding token ID.

  Returns:
    A tuple containing:
      - encoder_logits: Full-length logits ``[B, FS, V]``
        (``remove_mm_logits`` bypassed).
      - kv_cache: Prefilled KV cache (incl. image K/V) with set end index.
      - positions: Positional offset IDs.
      - prompt_mask: Mask filtering out pad tokens in prompt.
  """
  del total_canvas_len  # Unused; accepted for API consistency.
  # Concatenate prompt and clean canvas tokens.
  full_seq = jnp.concatenate([prompt, x0_tokens], axis=1)  # [B, FullSeqLen]
  # Mask out PAD tokens (the -2 soft-token slots are non-PAD -> valid).
  prompt_mask = prompt != pad_token  # [B, PromptLen]
  full_seq_mask = jnp.concatenate([prompt_mask, canvas_mask], axis=1)

  kv_cache, encoder_logits, positions, _ = (
      vision_hd_gemma_network.prefill_kv_cache_with_encoder(
          tokens=full_seq,
          input_mask=full_seq_mask,
          init_cache_fn=gemma_network.init_cache,
          encoder_fn=gemma_network.encoder_call,
          images=images,
      )
  )

  # Set end_index per example: prompt_len + selected_canvas_idx * canvas_size.
  # This makes the decoder reuse cached K/V for the prompt and all canvases
  # before the selected one.
  end_index = prompt_len + selected_canvas_idx * canvas_size  # [B]
  kv_cache = mask_helpers.set_cache_end_index(kv_cache, end_index)

  return encoder_logits, kv_cache, positions, prompt_mask


class VisionSFTDiffusion(sft_model.SFTDiffusion):
  """SFTDiffusion with image inputs threaded into the encoder prefill.

  Attributes:
    patches: Context key pointing to the image patches ``[B, P_p, p_d]``.
    positions_xy: Context key pointing to the patch positions ``[B, P_p, 2]``.
  """

  # Kontext keys (design §8: bound via "batch.patches"/"batch.positions_xy").
  patches: kd.kontext.Key = 'batch.patches'
  positions_xy: kd.kontext.Key = 'batch.positions_xy'

  @nn.compact
  def __call__(
      self,
      x0: jnp.ndarray,
      prompt: jnp.ndarray,
      canvas_id: jnp.ndarray,
      canvas_mask: jnp.ndarray,
      encoder_target: jnp.ndarray,
      encoder_target_mask: jnp.ndarray,
      patches: jnp.ndarray,
      positions_xy: jnp.ndarray,
      is_training: bool = True,
  ):
    """Computes model losses and forward predictions during training or eval.

    Identical to ``SFTDiffusion.__call__`` except that the encoder prefill
    receives ``images=(patches, positions_xy)`` (the only place vision enters
    the training step; design §3).

    Args:
      x0: Int array [B, SeqLen] of initial target canvas states.
      prompt: Int array [B, PromptLen] of input prompt sequences with the
        image span pre-expanded by the data pipeline.
      canvas_id: Int array [B, SeqLen] of segment identifier tokens.
      canvas_mask: Float/Int array [B, SeqLen] masking active (non-pad)
        canvas positions.
      encoder_target: Int array [B, SeqLen] of target token IDs.
      encoder_target_mask: Float/Int array [B, SeqLen] of target masks (image
        span already zeroed by the data pipeline).
      patches: Float array [B, P_p, p_d] of preprocessed image patches.
      positions_xy: Int array [B, P_p, 2] of patch grid positions (-1 = pad).
      is_training: If True, operates in training mode (enabling dropout,
        etc.).

    Returns:
      A dictionary containing logits, targets, predictions, and loss
      metadata.
    """
    ############################################################################
    # Sample time and corrupt x0
    ############################################################################

    # Sample time
    time = self.time_sampler(self.make_rng('sampling'), x0)

    # Corrupt x0 (the image is conditioning, not diffusion state — it is
    # never noised).
    xt, target_info = self.corruption_process.corrupt(
        self.make_rng('sampling'), x0, time
    )

    ############################################################################
    # Sample canvas
    ############################################################################

    # Sample which canvas to train on
    # Count valid canvases per example by checking the first token of each
    # canvas in canvas_mask.
    first_token_indices = jnp.arange(self.num_canvases) * self.canvas_size
    canvas_validity = canvas_mask[:, first_token_indices]  # [B, num_canvases]
    num_valid_canvases = jnp.sum(canvas_validity, axis=-1)  # [B]
    # Clip to at least 1 to avoid zero-division on empty examples.
    num_valid_canvases = jnp.maximum(num_valid_canvases, 1)

    # Uniformly sample a canvas index from [0, num_valid_canvases) per example.
    rng_canvas = self.make_rng('sampling')
    selected_canvas_idx = jax.random.randint(
        rng_canvas,
        shape=num_valid_canvases.shape,
        minval=0,
        maxval=num_valid_canvases,
    )  # [B]

    # Squeeze trailing dim if present (hackable diffusion uses <B, L, 1>).
    x0_tokens = x0[..., 0] if x0.ndim == 3 else x0  # [B, TotalCanvasLen]

    ############################################################################
    # Create KV cache and encoder logits (vision tower runs inside)
    ############################################################################

    encoder_logits, kv_cache, positions, prompt_mask = sft_encode(
        gemma_network=self.gemma_network,
        prompt=prompt,
        x0_tokens=x0_tokens,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        prompt_len=self.prompt_len,
        total_canvas_len=self.total_canvas_len,
        canvas_size=self.canvas_size,
        images=(patches, positions_xy),
        pad_token=self.pad_token,
    )

    if self.stop_gradient_from_denoiser_to_encoder:
      kv_cache = jax.lax.stop_gradient(kv_cache)

    ############################################################################
    # Decode first pass
    ############################################################################

    decoder_kwargs = dict(
        gemma_network=self.gemma_network,
        xt=xt,
        time=time,
        kv_cache=kv_cache,
        positions=positions,
        prompt_mask=prompt_mask,
        canvas_mask=canvas_mask,
        selected_canvas_idx=selected_canvas_idx,
        prompt_len=self.prompt_len,
        total_canvas_len=self.total_canvas_len,
        canvas_size=self.canvas_size,
        is_training=is_training,
    )

    denoiser_output_first_pass = sft_model.sft_decode(**decoder_kwargs)

    # Derive target_mask: only the selected canvas contributes to loss
    target_mask = canvas_mask & (canvas_id == selected_canvas_idx[:, None])

    # Combine is_corrupted with target_mask to ignore non-selected tokens
    target_info['is_corrupted'] = (
        target_info['is_corrupted'] & target_mask[..., None]
    )
    target_info['target_mask'] = target_mask[..., None]

    # Convert predictions (computes loss-ready dict)
    converted_first_pass = self.corruption_process.convert_predictions(
        denoiser_output_first_pass, xt, time
    )

    ############################################################################
    # Self-conditioning & decode second pass
    ############################################################################

    converted_first_pass = jax.lax.stop_gradient(converted_first_pass)
    sc_logits = converted_first_pass['logits']
    zero_logits = jnp.zeros_like(sc_logits)

    # With probability self_cond_prob, run self-conditioning element-wise.
    batch_size = xt.shape[0]
    do_self_cond = (
        jax.random.uniform(self.make_rng('sampling'), shape=(batch_size,))
        < self.self_cond_prob
    )
    # Reshape to broadcast with x0_hat_logits (Batch, ..., Channels)
    do_self_cond = do_self_cond.reshape(
        (batch_size,) + (1,) * (sc_logits.ndim - 1)
    )
    sc_logits = jnp.where(do_self_cond, sc_logits, zero_logits)

    denoiser_output = sft_model.sft_decode(**decoder_kwargs, sc_logits=sc_logits)

    # Convert predictions (computes loss-ready dict)
    converted = self.corruption_process.convert_predictions(
        denoiser_output, xt, time
    )

    # Get noise info
    noise_info = self.corruption_process.get_schedule_info(time)

    return {
        'output': converted,
        'target': target_info,
        'xt': xt,
        'noise_info': noise_info,
        'encoder_logits': encoder_logits,
        'encoder_target': encoder_target,
        'encoder_target_mask': encoder_target_mask,
    }


class VisionGemmaSamplingEvaluator(sft_model.GemmaSamplingEvaluator):
  """Sampling evaluator that adds the image tensors to the conditioning.

  Vision touches the sampling flow in exactly one place — the prompt prefill
  inside ``init_ar_state`` (design §6); here we only extend the conditioning
  dict with the patch tensors so the state handler can forward them.
  """

  def _ar_diffusion_step(
      self, step_nr: int, state: kd.train.TrainState, batch: Any
  ) -> kd.train.AuxiliariesState:
    """Runs full AR diffusion token generation with image conditioning.

    Args:
      step_nr: Current evaluation step number.
      state: Kauldron training state containing model parameters.
      batch: The evaluation data batch.

    Returns:
      Auxiliaries state containing generated samples and latency summaries.
    """
    # Set up the context and the inference function
    base_context = kd.train.Context.from_state_and_batch(
        state=state, batch=batch
    )
    context = sft_model.SamplingContext(**base_context.__dict__)
    inference_fn = self._make_inference_fn(self.model, context)
    # Update the context of the AR sampler.
    assert self.ar_diffusion_sampler is not None
    self.ar_diffusion_sampler.update_from_context(context)

    # Create PRNG keys for init and sampling
    rngs = self.base_cfg.rng_streams.eval_rngs(step_nr)
    _, sample_rng = jax.random.split(rngs[self.rng_stream], 2)

    _, kwargs = kd.data.utils.get_model_inputs(self.model, context)
    prompt_tokens = kwargs['prompt']
    # The prompt arrives from the eval pipeline already expanded (-2 slots
    # included), exactly like training; the -2 slots are non-PAD so they are
    # counted as real prompt tokens.
    prompt_lengths = jnp.sum(prompt_tokens != self.pad_token, axis=-1)  # [B]

    cond = {
        'prompt_tokens': prompt_tokens,
        'prompt_lengths': prompt_lengths,
        'patches': kwargs['patches'],
        'positions_xy': kwargs['positions_xy'],
    }

    # Run the sampling loop
    final, final_state = self.ar_diffusion_sampler(
        diffusion_inference_fn=inference_fn,
        batch_size=len(prompt_tokens),
        rng=sample_rng,
        conditioning=cond,
    )
    final = jnp.expand_dims(final, axis=-1)

    processed_denoising_steps = jnp.array(
        final_state['processed_denoising_steps'], dtype=jnp.float32
    ).reshape(())
    processed_num_canvases = jnp.array(
        final_state['processed_num_canvases'], dtype=jnp.float32
    ).reshape(())
    average_denoising_steps_per_canvas = (
        processed_denoising_steps / processed_num_canvases
    )
    average_denoising_steps_per_canvas = jnp.array(
        average_denoising_steps_per_canvas, dtype=jnp.float32
    ).reshape(())

    # Update the context with the final and intermediate samples
    # final and interms are DiffusionStep trees.
    context = context.replace(
        samples=final,
        # Latency metrics.
        processed_denoising_steps=processed_denoising_steps,
        processed_num_canvases=processed_num_canvases,
        average_denoising_steps_per_canvas=average_denoising_steps_per_canvas,
    )
    # Compute the metrics
    context = self.aux.update_context(context)
    return context.get_aux_state(
        return_losses=True, return_metrics=True, return_summaries=True
    )

  def __hash__(self) -> int:
    # Make Evaluator hashable, so its methods can be jitted.
    return id(self)
