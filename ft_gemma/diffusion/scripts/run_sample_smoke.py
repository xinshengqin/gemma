"""Inference/sampling smoke run + design shape/mask verification.

Invoked by ``sample_sudoku_vision_smoke.sh``. Counterpart of
``run_train_smoke.py`` for the inference side (design §6 "Inference
sequence" / §11 "Life of a Sample"):

1. Resolves ``configs/sft_sudoku_vision_full_tiny.py``, initializes the
   parameters (no checkpoint needed — the smoke validates machinery, not
   model quality), and builds the sampling stack exactly as the evaluators
   do: ``VisionGemmaARStateHandler`` + ``GemmaKDARSampler`` +
   ``SFTInferenceFn``.

2. Verifies the prompt-prefill state (``init_ar_state``) against the
   design: cache C = P + NC*CS with the write cursor at P, canvas RoPE
   positions, and the decoder attention mask [B, CS, C] — image soft-token
   keys admitted, prompt PAD hidden, canvas-0 block open.

3. Runs the FULL autoregressive diffusion sampling loop (outer canvas loop,
   inner DDIM denoising loop, stop-token truncation, finalize) and verifies
   the outputs: samples [B, NC*CS] with valid token ids and the latency
   counters.
"""

from absl import app
from absl import flags

_DENOISING_STEPS = flags.DEFINE_integer(
    "denoising_steps", 4, "Inner diffusion loop steps T_s (tiny smoke value)."
)

# Design constants (docs/add-visual-inputs/index.html §9, tiny smoke config).
B = 2  # batch size
P = 384  # padded prompt length
CS = 256  # canvas size
NC = 1  # num canvases
C = P + NC * CS  # KV-cache length = 640
V = 262_144  # vocab size
P_P = 2520  # max patches per image
P_D = 768  # patch dim
N_KV, D_H = 2, 8  # tiny local KV heads / head dim
N_GKV, K_G = 1, 16  # tiny global KV heads / key size


def _check(name, actual, expected):
  assert tuple(actual) == tuple(expected), (
      f"{name}: got {tuple(actual)}, design says {tuple(expected)}"
  )
  print(f"  OK  {name:<46} {tuple(actual)}")


def main(argv):
  del argv
  # Imports deferred so absl flags parse before jax/tf initialisation noise.
  import jax
  import jax.numpy as jnp
  import numpy as np
  from hackable_diffusion import hd
  from kauldron import konfig

  from ft_gemma.diffusion.hackable_diffusion_adapter import compat
  from ft_gemma.diffusion.hackable_diffusion_adapter.configs import (
      sft_sudoku_vision_full_tiny,
  )
  from ft_gemma.diffusion.hackable_diffusion_adapter.hd import (
      vision_ar_state_handler,
  )
  from ft_gemma.diffusion.hackable_diffusion_adapter.hd import vision_sft_model
  from gemma import gm
  from gemma.diffusion.hackable_diffusion_adapter.hd import (
      hd_gemma_ar_state_handler,
  )

  compat.patch_etils_jax_prng()

  cfg = sft_sudoku_vision_full_tiny.get_config()
  trainer = konfig.resolve(cfg)

  ##############################################################################
  # 1. Build the sampling stack from freshly initialized parameters.
  ##############################################################################
  print("== initializing parameters and the sampling stack ==")
  state = trainer.init_state()
  gemma_network = trainer.model.gemma_network
  gemma_network_params = state.params["gemma_network"]

  batch = next(iter(trainer.eval_ds))
  prompt_tokens = jnp.asarray(batch["prompt"])
  prompt_lengths = jnp.sum(prompt_tokens != 0, axis=-1)
  # The conditioning dict, exactly as VisionGemmaSamplingEvaluator builds it.
  cond = {
      "prompt_tokens": prompt_tokens,
      "prompt_lengths": prompt_lengths,
      "patches": jnp.asarray(batch["patches"]),
      "positions_xy": jnp.asarray(batch["positions_xy"]),
  }
  _check("cond.prompt_tokens [B, P]", cond["prompt_tokens"].shape, (B, P))
  _check("cond.patches [B, P_p, p_d]", cond["patches"].shape, (B, P_P, P_D))
  _check(
      "cond.positions_xy [B, P_p, 2]", cond["positions_xy"].shape, (B, P_P, 2)
  )

  special = gm.text.Gemma4Tokenizer.special_tokens
  handler = vision_ar_state_handler.VisionGemmaARStateHandler(
      gemma_network=gemma_network,
      gemma_params=gemma_network_params["gemma_model"],
      pad_token=special.PAD,
      end_tokens=(
          special.EOS,
          special.END_OF_TURN,
          special.BEGIN_OF_TOOL_RESPONSE,
      ),
  )
  corruption_process = hd.corruption.CategoricalProcess.uniform_process(
      num_categories=V,
      schedule=hd.corruption.RFSchedule(),
  )
  sampler = vision_sft_model.GemmaKDARSampler(
      state_handler=handler,
      canvas_sampler=hd.sampling.DiffusionSampler(
          time_schedule=hd.sampling.UniformTimeSchedule(),
          stepper=hd.sampling.DiscreteDDIMStep(
              corruption_process=corruption_process,
              temperature=0.7,
              logits_dtype=jnp.bfloat16,
          ),
          update_conditioning_fn=hd_gemma_ar_state_handler.PropagateSelfConditioningFn(),
          num_steps=_DENOISING_STEPS.value,
          store_trajectory=False,
      ),
      diffusion_process=corruption_process,
      canvas_length=CS,
      max_num_canvases=NC,
      data_dtype=jnp.int32,
      data_shape=(1,),
  )
  inference_fn = vision_sft_model.SFTInferenceFn(
      gemma_network=gemma_network,
      params=gemma_network_params,
  )

  ##############################################################################
  # 2. Prompt prefill (init_ar_state) — design §11 step 2.
  ##############################################################################
  print("== verifying the prompt prefill state against the design ==")
  init_state = handler.init_ar_state(
      batch_size=B,
      conditioning=dict(cond),
      canvas_length=CS,
      max_num_canvases=NC,
  )

  kv = init_state["kv_cache"]
  _check(
      "cache k local layer [B, C, N_kv, d_h]",
      kv["layer_0"]["k"].shape,
      (B, C, N_KV, D_H),
  )
  _check(
      "cache k global layer [B, C, N_gkv, K_g]",
      kv["layer_2"]["k"].shape,
      (B, C, N_GKV, K_G),
  )
  np.testing.assert_array_equal(np.asarray(kv["layer_0"]["end_index"]), [P] * B)
  print(f"  OK  cache write cursor (end_index) == P == {P}")

  _check(
      "canvas positions [B, CS]", init_state["positions"].shape, (B, CS)
  )
  np.testing.assert_array_equal(
      np.asarray(init_state["positions"]),
      np.asarray(prompt_lengths)[:, None] + np.arange(CS)[None, :],
  )
  print("  OK  canvas positions == prompt_lengths + arange(CS)")

  _check(
      "decoder attention mask [B, CS, C]",
      init_state["attention_mask"].shape,
      (B, CS, C),
  )
  # Mask contents (design §11): every canvas query sees the non-PAD prompt
  # keys — the -2 image soft-token slots included — plus all of canvas 0.
  prompt_np = np.asarray(prompt_tokens)
  expected_row = np.concatenate(
      [prompt_np != 0, np.ones((B, CS), dtype=bool)], axis=1
  )  # [B, C]
  np.testing.assert_array_equal(
      np.asarray(init_state["attention_mask"]),
      np.broadcast_to(expected_row[:, None, :], (B, CS, C)),
  )
  soft_cols = prompt_np[0] == -2
  assert np.asarray(init_state["attention_mask"])[0, 0, :P][soft_cols].all()
  print("  OK  decoder mask admits image keys, hides prompt PAD")

  np.testing.assert_array_equal(
      np.asarray(init_state["predicted_tokens"][:, :P]), prompt_np
  )
  print("  OK  output buffer starts with the prompt; canvas slots empty")

  ##############################################################################
  # 3. Full AR diffusion sampling loop — design §11 steps 3-4.
  ##############################################################################
  steps = _DENOISING_STEPS.value
  print(f"== running the full AR sampling loop (T_s={steps}) ==")
  samples, final_state = sampler(
      diffusion_inference_fn=inference_fn,
      batch_size=B,
      rng=jax.random.PRNGKey(0),
      conditioning=dict(cond),
  )

  _check("samples [B, NC*CS]", samples.shape, (B, NC * CS))
  samples_np = np.asarray(samples)
  assert samples_np.dtype.kind == "i", f"samples dtype {samples_np.dtype}"
  assert (samples_np >= 0).all() and (samples_np < V).all(), (
      "sampled token ids outside [0, V)"
  )
  print(f"  OK  sampled token ids within [0, {V})")

  num_canvases = int(np.asarray(final_state["processed_num_canvases"]))
  assert num_canvases == NC, f"processed {num_canvases} canvases, expected {NC}"
  print(f"  OK  processed_num_canvases == {NC}")
  denoising_steps = int(np.asarray(final_state["processed_denoising_steps"]))
  assert denoising_steps > 0
  print(f"  OK  processed_denoising_steps == {denoising_steps}")

  np.testing.assert_array_equal(
      np.asarray(final_state["predicted_tokens"][:, :P]), prompt_np
  )
  print("  OK  prompt prefix preserved in the output buffer")

  print("SAMPLING SMOKE TEST PASSED")


if __name__ == "__main__":
  app.run(main)
