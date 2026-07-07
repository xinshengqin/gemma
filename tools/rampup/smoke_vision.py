"""Smoke verification for the add-visual-inputs design docs.

Measures the *existing* gemma4 vision components that the design reuses
verbatim (the designed adapter/data changes do not exist yet and are marked
[DESIGN] in the docs):

  1. Full 26B_A4B vision tower + mm projection weight inventory via
     jax.eval_shape (no memory needed) — exact shapes and param counts.
  2. A tiny VisionEncoder forward: preprocess -> patchify_and_pad ->
     VisionEncoder -> Embedder.encode_vision, recording every I/O shape.
  3. Variable-aspect-ratio soft-token counts for a few image sizes.
  4. add_variable_extra_tokens_for_images expansion on a toy prompt.
  5. merge_flat_embeddings scatter semantics (incl. the pad-to-slot-0 trick).

Run:  PYTHONPATH=<repo root> python tools/rampup/smoke_vision.py
Outputs: tools/rampup/measured/vision_*.json
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from gemma.gm.nn.gemma4 import _modules
from gemma.gm.nn.gemma4.vision import _encoder as gemma4_vision
from gemma.gm.nn.gemma4.vision import _preprocessing
from gemma.gm.vision import _token_utils

OUT_DIR = os.path.join(os.path.dirname(__file__), "measured")
os.makedirs(OUT_DIR, exist_ok=True)


def _tree_shapes(params, prefix=""):
  out = {}
  for k, v in params.items():
    path = f"{prefix}/{k}" if prefix else k
    if isinstance(v, dict):
      out.update(_tree_shapes(v, path))
    else:
      out[path] = {
          "shape": list(v.shape),
          "dtype": str(v.dtype),
          "count": int(np.prod(v.shape)) if v.shape else 1,
      }
  return out


# ---------------------------------------------------------------------------
# 1. Full 26B vision tower + mm projection inventory (eval_shape only).
# ---------------------------------------------------------------------------
def full_inventory():
  # Exact config from Gemma4_26B_A4B (_gemma4.py:286-294).
  enc = gemma4_vision.VisionEncoder(
      d_model=1152,
      num_layers=27,
      num_heads=16,
      ffw_hidden=4304,
      output_length=280,
      use_clipped_linears=False,
      standardize_embeddings=True,
  )
  max_patches = enc.max_patches  # 280 * 3**2 = 2520
  patch_dim = enc.patch_size * enc.patch_size * 3  # 768

  def init_fn():
    patches = jnp.zeros((1, max_patches, patch_dim), jnp.float32)
    pos = jnp.zeros((1, max_patches, 2), jnp.int32)
    return enc.init(jax.random.PRNGKey(0), patches, pos)

  vars_shape = jax.eval_shape(init_fn)
  tower = _tree_shapes(vars_shape["params"])

  # mm projection lives in the LM Embedder (_modules.py:88-92).
  emb = _modules.Embedder(
      vocab_size=262_144,
      embed_dim=2816,
      vision_proj_dim=1152,
  )

  def init_emb():
    return emb.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, 1, 4, 1152), jnp.float32),
        method=emb.encode_vision,
    )

  emb_shape = jax.eval_shape(init_emb)
  proj = {
      k: v
      for k, v in _tree_shapes(emb_shape["params"]).items()
      if k.startswith("mm_")
  }

  tower_total = sum(v["count"] for v in tower.values())
  proj_total = sum(v["count"] for v in proj.values())
  result = {
      "vision_encoder_params": tower,
      "vision_encoder_total": tower_total,
      "mm_projection_params": proj,
      "mm_projection_total": proj_total,
      "grand_total_new_params": tower_total + proj_total,
      "max_patches": max_patches,
      "patch_dim": patch_dim,
      "num_mm_tokens_per_image": enc.num_mm_tokens_per_image,
      "image_budget_hw": [enc.image_height, enc.image_width],
  }
  with open(os.path.join(OUT_DIR, "vision_weights_full.json"), "w") as f:
    json.dump(result, f, indent=2)
  print(
      f"[1] tower={tower_total:,}  proj={proj_total:,}  "
      f"total_new={tower_total + proj_total:,}"
  )
  return result


# ---------------------------------------------------------------------------
# 2. Tiny VisionEncoder end-to-end forward with recorded module I/O.
# ---------------------------------------------------------------------------
def tiny_forward():
  io_log = {}
  tiny = gemma4_vision.VisionEncoder(
      d_model=16,
      num_layers=2,
      num_heads=2,
      ffw_hidden=32,
      output_length=280,  # max budget; real count comes from image size
      standardize_embeddings=True,
  )
  # Variable aspect ratio image: 120x200 -> resized to multiples of 48.
  rng = np.random.RandomState(0)
  img = rng.randint(0, 255, (120, 200, 3), dtype=np.uint8)
  pre = _preprocessing.preprocess_image(img, max_soft_tokens=280)
  io_log["preprocess_image"] = {
      "in": list(img.shape),
      "out": list(pre.shape),
      "note": "resized to multiples of 48px, aspect preserved, [0,1] f32",
  }
  patches, positions_xy, n_real = gemma4_vision.patchify_and_pad(
      [pre], max_soft_tokens=280
  )
  io_log["patchify_and_pad"] = {
      "patches": list(patches.shape),
      "positions_xy": list(positions_xy.shape),
      "num_real_patches_per_image": [int(x) for x in n_real],
  }
  s_v = int(n_real[0]) // 9
  io_log["soft_tokens_S_v"] = s_v

  variables = tiny.init(jax.random.PRNGKey(0), patches, positions_xy)
  weights = _tree_shapes(variables["params"])

  intercepted = []

  def interceptor(next_fun, args, kwargs, context):
    out = next_fun(*args, **kwargs)

    def shp(x):
      return list(x.shape) if hasattr(x, "shape") else type(x).__name__

    intercepted.append({
        "module": type(context.module).__name__,
        "method": context.method_name,
        "out": jax.tree.map(shp, out),
    })
    return out

  from flax import linen as nn

  with nn.intercept_methods(interceptor):
    outputs = tiny.apply(variables, patches, positions_xy)
  (embeddings, mask) = outputs[0]
  io_log["vision_encoder_out"] = {
      "embeddings": list(embeddings.shape),
      "mask": list(mask.shape) if mask is not None else None,
      "valid_soft_tokens": int(mask.sum()) if mask is not None else None,
  }

  # Projection into a tiny LM space (D=32 like the original tiny smoke run).
  emb = _modules.Embedder(vocab_size=64, embed_dim=32, vision_proj_dim=16)
  ev = embeddings[0][jnp.nonzero(mask[0], size=s_v)[0]][None, None]
  emb_vars = emb.init(jax.random.PRNGKey(1), ev, method=emb.encode_vision)
  projected = emb.apply(emb_vars, ev, method=emb.encode_vision)
  io_log["encode_vision"] = {"in": list(ev.shape), "out": list(projected.shape)}

  result = {
      "tiny_weights": weights,
      "tiny_total": sum(v["count"] for v in weights.values()),
      "io": io_log,
      "module_calls": intercepted,
  }
  with open(os.path.join(OUT_DIR, "vision_tiny_io.json"), "w") as f:
    json.dump(result, f, indent=2)
  print(f"[2] tiny tower total={result['tiny_total']:,}  S_v={s_v}")
  return result


# ---------------------------------------------------------------------------
# 3. Aspect-ratio table.
# ---------------------------------------------------------------------------
def aspect_table():
  rows = []
  for h, w in [(800, 800), (1080, 1920), (120, 200), (2000, 500), (48, 48)]:
    th, tw = _preprocessing.get_target_dimensions(h, w, max_patches=2520)
    s_v = (th // 16) * (tw // 16) // 9
    rows.append({"in_hw": [h, w], "resized_hw": [th, tw], "soft_tokens": s_v})
  with open(os.path.join(OUT_DIR, "vision_aspect_ratios.json"), "w") as f:
    json.dump(rows, f, indent=2)
  print("[3]", rows)
  return rows


# ---------------------------------------------------------------------------
# 4. Token expansion.
# ---------------------------------------------------------------------------
def expansion():
  from gemma.gm.text import _tokenizer

  st = _tokenizer.Gemma4Tokenizer.special_tokens
  toks = np.array([[2, 10, 11, st.IMAGE_PLACEHOLDER, 12]], np.int32)
  out = _token_utils.add_variable_extra_tokens_for_images(
      toks, soft_token_counts=[5]
  )
  result = {
      "special_tokens": {
          "IMAGE_PLACEHOLDER": st.IMAGE_PLACEHOLDER,
          "START_OF_IMAGE": st.START_OF_IMAGE,
          "END_OF_IMAGE": st.END_OF_IMAGE,
          "SOFT_TOKEN_PLACEHOLDER": _token_utils.SOFT_TOKEN_PLACEHOLDER,
      },
      "in": toks.tolist(),
      "out": out.tolist(),
      "in_len": toks.shape[1],
      "out_len": out.shape[1],
      "span_per_image(S_v=5)": out.shape[1] - toks.shape[1] + 1,
  }
  with open(os.path.join(OUT_DIR, "vision_token_expansion.json"), "w") as f:
    json.dump(result, f, indent=2)
  print(f"[4] expansion {toks.shape[1]} -> {out.shape[1]} (S_v=5 => span 9)")
  return result


# ---------------------------------------------------------------------------
# 5. merge_flat_embeddings semantics (batched, variable counts).
# ---------------------------------------------------------------------------
def merge_semantics():
  b, l, t, d = 2, 8, 4, 3
  text = jnp.arange(b * l * d, dtype=jnp.float32).reshape(b, l, d)
  mm = -jnp.ones((b, t, d), jnp.float32)
  # Example 0 has 4 placeholders, example 1 only 2 (variable counts).
  mask = jnp.array([
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 0, 0, 0, 0],
  ], dtype=bool)
  merged = _token_utils.merge_flat_embeddings(
      text_embeddings=text, multimodal_embeddings=mm, mask=mask
  )
  result = {
      "shapes": {"text": [b, l, d], "mm": [b, t, d], "mask": [b, l]},
      "merged_row1_pos0_preserved": bool(
          jnp.allclose(merged[1, 0], text[1, 0])
      ),
      "merged_row1_pos2_is_mm": bool(jnp.allclose(merged[1, 2], -1.0)),
      "note": (
          "excess mm rows scatter to slot 0 then slot 0 is restored -> "
          "per-example variable counts are safe if T >= per-example count"
      ),
  }
  with open(os.path.join(OUT_DIR, "vision_merge_semantics.json"), "w") as f:
    json.dump(result, f, indent=2)
  print("[5]", result["merged_row1_pos0_preserved"],
        result["merged_row1_pos2_is_mm"])
  return result


if __name__ == "__main__":
  full_inventory()
  tiny_forward()
  aspect_table()
  expansion()
  merge_semantics()
  print("done ->", OUT_DIR)
