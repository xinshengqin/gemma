"""One-step training smoke run + design shape/mask verification.

Invoked by ``train_sudoku_vision_smoke.sh``. Does two things:

1. Resolves ``configs/sft_sudoku_vision_full_tiny.py`` and runs the existing
   Kauldron trainer for its configured single step — one forward pass, one
   backward pass, and one optimizer update — asserting that the parameters
   (including the vision tower and mm projection) actually changed.

2. Re-derives the tensors of one training step from the trained state and
   checks every shape and the attention-mask contents against the design
   walkthrough (docs/add-visual-inputs/index.html §5.1, §9, §10):
     - batch fields, encoder/denoiser logits, KV cache (local + global);
     - causal prefill mask (strictly causal, PAD keys hidden — GLOBAL
       layers; image tokens causal there);
     - vision sliding mask (bidirectional exactly within the image block,
       causal outside — LOCAL_SLIDING layers);
     - decoder mask (image keys admitted as non-PAD prompt positions, PAD
       hidden, canvases <= selected visible).
"""

import os

from absl import app
from absl import flags

_WORKDIR = flags.DEFINE_string(
    "workdir", "/tmp/ft_gemma_sudoku_vision_smoke", "Trainer workdir."
)

# Design constants (docs/add-visual-inputs/index.html §9, tiny smoke config).
B = 2  # batch size
P = 384  # padded prompt length
CS = 256  # canvas size
NC = 1  # num canvases
TC = NC * CS  # total canvas length
FS = P + TC  # full sequence = KV-cache length C
V = 262_144  # vocab size
P_P = 2520  # max patches per image = S_max * 3**2
P_D = 768  # patch dim = 16*16*3
S_MAX = 280  # padded soft tokens per image
N_KV, D_H = 2, 8  # tiny local KV heads / head dim
N_GKV, K_G = 1, 16  # tiny global KV heads / key size


def _check(name, actual, expected):
  assert tuple(actual) == tuple(expected), (
      f"{name}: got {tuple(actual)}, design says {tuple(expected)}"
  )
  print(f"  OK  {name:<42} {tuple(actual)}")


def main(argv):
  del argv
  # Imports deferred so absl flags parse before jax/tf initialisation noise.
  import jax
  import jax.numpy as jnp
  import numpy as np
  from kauldron import konfig

  from ft_gemma.diffusion.hackable_diffusion_adapter import compat

  compat.patch_etils_jax_prng()

  from ft_gemma.diffusion.hackable_diffusion_adapter.configs import (
      sft_sudoku_vision_full_tiny,
  )
  from ft_gemma.diffusion.hackable_diffusion_adapter.hd import (
      vision_mask_helpers,
  )
  from gemma.diffusion.hackable_diffusion_adapter.hd import mask_helpers

  cfg = sft_sudoku_vision_full_tiny.get_config()
  cfg.workdir = _WORKDIR.value
  trainer = konfig.resolve(cfg)

  ##############################################################################
  # 1. One training step: forward + backward + optimizer update.
  ##############################################################################
  print("== training for 1 step (forward + backward + optimizer update) ==")
  init_state = trainer.init_state()
  params_before = jax.tree.map(np.asarray, init_state.params)
  state, _ = trainer.train()
  assert int(state.step) == 1, f"expected 1 optimizer step, got {state.step}"

  # The update must have changed the params — including the fp32 vision
  # tower and the mm projection (trainable per the design).
  changed = jax.tree.map(
      lambda a, b: bool(np.any(np.asarray(a) != np.asarray(b))),
      params_before,
      jax.tree.map(np.asarray, state.params),
  )
  flat = {
      jax.tree_util.keystr(k): v
      for k, v in jax.tree_util.tree_flatten_with_path(changed)[0]
  }
  n_changed = sum(flat.values())
  print(f"  params updated: {n_changed}/{len(flat)} tensors changed")
  assert n_changed > 0, "optimizer update did not change any parameter"
  for pattern in ("vision_encoder", "mm_input_projection"):
    assert any(v for k, v in flat.items() if pattern in k), (
        f"no {pattern} parameter changed — vision tower must be trainable"
    )
  print("  OK  vision tower + mm projection received an update")

  ##############################################################################
  # 2. Shape verification against the design.
  ##############################################################################
  print("== verifying tensor shapes against the design ==")
  batch = next(iter(trainer.train_ds))
  _check("batch.prompt [B, P]", batch["prompt"].shape, (B, P))
  _check("batch.patches [B, P_p, p_d]", batch["patches"].shape, (B, P_P, P_D))
  _check(
      "batch.positions_xy [B, P_p, 2]", batch["positions_xy"].shape, (B, P_P, 2)
  )
  _check("batch.canvas [B, TC, 1]", batch["canvas"].shape, (B, TC, 1))
  _check("batch.canvas_id [B, TC]", batch["canvas_id"].shape, (B, TC))
  _check("batch.canvas_mask [B, TC]", batch["canvas_mask"].shape, (B, TC))
  _check(
      "batch.encoder_target [B, FS]", batch["encoder_target"].shape, (B, FS)
  )
  _check(
      "batch.encoder_target_mask [B, FS]",
      batch["encoder_target_mask"].shape,
      (B, FS),
  )

  prompt = jnp.asarray(batch["prompt"])
  canvas_mask = jnp.asarray(batch["canvas_mask"])
  x0_tokens = jnp.asarray(batch["canvas"])[..., 0]
  patches = jnp.asarray(batch["patches"])
  positions_xy = jnp.asarray(batch["positions_xy"])

  full_seq = jnp.concatenate([prompt, x0_tokens], axis=1)
  prompt_mask = prompt != 0
  full_seq_mask = jnp.concatenate([prompt_mask, canvas_mask], axis=1)

  # Encoder prefill masks (built exactly as sft_encode builds them).
  attn_mask, sliding_mask = vision_mask_helpers.make_vision_prefill_masks(
      tokens=full_seq, token_mask=full_seq_mask, cache_length=FS
  )
  _check("causal prefill mask [B, FS, FS]", attn_mask.shape, (B, FS, FS))
  _check("vision sliding mask [B, FS, FS]", sliding_mask.shape, (B, FS, FS))

  # Decoder mask (as sft_decode builds it, selected canvas 0).
  dec_mask = mask_helpers.create_decoder_attention_mask(
      prompt_mask=prompt_mask,
      canvas_mask=canvas_mask,
      selected_canvas_idx=jnp.zeros((B,), jnp.int32),
      prompt_len=P,
      total_canvas_len=TC,
      canvas_size=CS,
      num_queries=TC,
  )
  _check("decoder attention mask [B, TC, FS]", dec_mask.shape, (B, TC, FS))

  # Encoder + denoiser forward, using the trained params, to verify the
  # network-side shapes (logits, cache) — same calls sft_encode/sft_decode
  # make.
  gemma_network = trainer.model.gemma_network
  gn_params = state.params["gemma_network"]

  cache = gemma_network.apply(
      {"params": gn_params},
      batch_size=B,
      cache_length=FS,
      method=gemma_network.init_cache,
  )
  positions = mask_helpers.build_positions_from_mask(full_seq_mask)
  encoder_out = gemma_network.apply(
      {"params": gn_params},
      x=full_seq,
      conditioning_embeddings={
          "kv_cache": cache,
          "positions": positions,
          "attention_mask": attn_mask,
          "sliding_attention_mask": sliding_mask,
          "images": (patches, positions_xy),
      },
      method=gemma_network.encoder_call,
  )
  _check(
      "encoder_logits [B, FS, V] (remove_mm_logits bypassed)",
      encoder_out.logits.shape,
      (B, FS, V),
  )
  kv = encoder_out.cache
  _check(
      "cache k local layer [B, C, N_kv, d_h]",
      kv["layer_0"]["k"].shape,
      (B, FS, N_KV, D_H),
  )
  _check(
      "cache k global layer [B, C, N_gkv, K_g]",
      kv["layer_2"]["k"].shape,
      (B, FS, N_GKV, K_G),
  )
  _check("cache end_index [B]", kv["layer_0"]["end_index"].shape, (B,))
  assert int(kv["layer_0"]["end_index"][0]) == FS

  kv = mask_helpers.set_cache_end_index(
      kv, jnp.full((B,), P, dtype=jnp.int32)
  )
  denoiser_out = gemma_network.apply(
      {"params": gn_params},
      xt=jnp.asarray(batch["canvas"]),
      time=jnp.ones((B, 1, 1), jnp.float32),
      conditioning={
          "kv_cache": kv,
          "positions": positions[:, P:],
          "attention_mask": dec_mask,
      },
      is_training=False,
  )
  _check(
      "denoiser logits [B, TC, V]", denoiser_out["logits"].shape, (B, TC, V)
  )

  ##############################################################################
  # 3. Attention-mask content verification against the design.
  ##############################################################################
  print("== verifying attention-mask contents against the design ==")
  am = np.asarray(attn_mask[0])
  sm = np.asarray(sliding_mask[0])
  dm = np.asarray(dec_mask[0])
  p0 = np.asarray(prompt[0])
  valid = np.asarray(full_seq_mask[0])

  soft = np.nonzero(p0 == -2)[0]
  first_soft, last_soft = int(soft[0]), int(soft[-1])
  span_start, span_end = first_soft - 2, last_soft + 2  # \n\n <soi> .. <eoi> \n\n
  prompt_len_real = int((p0 != 0).sum())
  pad_cols = np.nonzero(~valid)[0]
  print(
      f"  image span: [{span_start}, {span_end}] "
      f"(S_v={len(soft)}, span={len(soft) + 4} tokens), "
      f"real prompt len={prompt_len_real}"
  )

  q, k_in, k_after = first_soft + 1, first_soft, last_soft  # probes
  tri = np.tril(np.ones((FS, FS), bool))

  # (1) Causal prefill mask: mask[q, k] = (k <= q) AND valid[k] — image
  # tokens strictly causal (this is what the GLOBAL layers consume).
  assert (am == (tri & valid[None, :])).all(), "causal mask != tril & valid"
  assert not am[q, k_after], "global-layer mask must stay causal in the image"
  print("  OK  causal prefill mask == lower-triangular AND token-valid")

  # (2) Sliding mask: identical to the causal mask outside the image block;
  # fully mutual (bidirectional) inside it (LOCAL_SLIDING layers).
  img = np.zeros(FS, bool)
  img[soft] = True
  bidir = img[:, None] & img[None, :]
  assert (sm == (am | bidir)).all(), (
      "sliding mask != causal mask OR image-block-bidirectional"
  )
  assert sm[q, k_after] and sm[first_soft, last_soft], (
      "soft tokens must see each other bidirectionally in the sliding mask"
  )
  assert not sm[q, span_end + 1], "no lookahead past the image block"
  assert not sm[span_start - 1, first_soft], (
      "text before the image must not see the soft tokens"
  )
  print("  OK  sliding mask bidirectional exactly within the image block")

  # (3) PAD keys hidden from every query, in both masks.
  assert not am[:, pad_cols].any() and not sm[:, pad_cols].any()
  print("  OK  PAD keys hidden from all prefill queries")

  # (4) Decoder mask: every canvas query row is identical (broadcast); it
  # admits all non-PAD prompt keys — image span included — plus the keys of
  # canvases with id <= selected; PAD hidden.
  expected_row = np.zeros(FS, bool)
  expected_row[:P] = p0 != 0
  expected_row[P:] = np.asarray(canvas_mask[0])  # canvas 0 selected, NC=1
  assert (dm == expected_row[None, :]).all(), "decoder mask row mismatch"
  assert dm[0, first_soft] and dm[0, span_start] and dm[0, span_end], (
      "decoder mask must admit the image keys (non-PAD prompt positions)"
  )
  assert not dm[0, prompt_len_real : P].any(), "prompt PAD keys must be hidden"
  print("  OK  decoder mask admits image keys, hides PAD, canvases <= selected")

  print("SMOKE TEST PASSED")


if __name__ == "__main__":
  app.run(main)
