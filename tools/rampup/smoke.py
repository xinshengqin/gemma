# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke run for the DiffusionGemma SFT (sft_sudoku_full.py) rampup docs.

Builds a tiny DiffusionGemma with the same topology as DiffusionGemma_26B_A4B
(local-sliding + global attention, MoE + dense-shared FFN, self-conditioning),
then:
  1. runs one SFTDiffusion training forward + both config losses + backward +
     one step of the config's optax chain,
  2. runs one AR-diffusion sampling call (GemmaKDARSampler, DiscreteDDIMStep),
  3. records every flax module call's input/output shapes via
     nn.intercept_methods,
  4. dumps the tiny weight inventory and the FULL 26B_A4B weight inventory
     (via jax.eval_shape - no memory needed),
to JSON files under tools/rampup/measured/.

Run from repo root:  python tools/rampup/smoke.py
"""

import json
import os
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from gemma.diffusion import _models
from gemma.diffusion import _transformer as diffusion_transformer
from gemma.gm.nn.gemma4 import _config
from gemma.gm.nn.gemma4 import _modules
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_ar_state_handler
from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model
from gemma.diffusion.hackable_diffusion_adapter.data import data as data_transforms
from hackable_diffusion import hd
from hackable_diffusion.lib.training import discrete_loss

OUT_DIR = os.path.join(os.path.dirname(__file__), 'measured')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Tiny config. Mirrors DiffusionGemma_26B_A4B topology (gemma4/_gemma4.py:258)
# with every dimension shrunk. 3 layers = 2x LOCAL_SLIDING + 1x GLOBAL so both
# attention variants and the MoE FFN are exercised.
# ---------------------------------------------------------------------------
TINY = dict(
    V=64,        # num_embed (26B: 262144)
    D=32,        # embed_dim (26B: 2816)
    NH=4,        # num_heads (26B: 16)
    NKV=2,       # num_kv_heads, local (26B: 8)
    HD=8,        # head_dim (26B: 256)
    NGKV=1,      # num_global_kv_heads (26B: 2)
    KG=16,       # global_key_size (26B: 512)
    HF=48,       # hidden_dim = moe_dense_hidden_dim (26B: 2112)
    E=4,         # num_experts (26B: 128)
    TOPK=2,      # top_k_experts (26B: 8)
    ED=16,       # expert_dim (26B: 704)
)
B = 2
PROMPT_LEN = 8
CANVAS_SIZE = 8
NUM_CANVASES = 2        # config uses 1; 2 exercises the general machinery
TOTAL_CANVAS_LEN = NUM_CANVASES * CANVAS_SIZE   # 16
FULL_SEQ_LEN = PROMPT_LEN + TOTAL_CANVAS_LEN    # 24
PAD, EOS = 0, 1
NUM_DENOISE_STEPS = 4

tiny_config = _config.TransformerConfig(
    num_embed=TINY['V'],
    embed_dim=TINY['D'],
    hidden_dim=TINY['HF'],
    num_heads=TINY['NH'],
    head_dim=TINY['HD'],
    num_kv_heads=TINY['NKV'],
    final_logit_softcap=30.0,
    num_global_kv_heads=TINY['NGKV'],
    use_post_attn_norm=True,
    use_post_ffw_norm=True,
    qk_norm_with_scale=True,
    attention_types=[
        _modules.AttentionType.LOCAL_SLIDING,
        _modules.AttentionType.LOCAL_SLIDING,
        _modules.AttentionType.GLOBAL,
    ],
    global_key_size=TINY['KG'],
    k_eq_v_global=True,
    global_rope_proportion=0.25,
    local_rope_proportion=1.0,
    attn_logits_soft_cap=None,
    sliding_window_size=8,
    local_base_frequency=10_000,
    global_base_frequency=1_000_000,
    per_layer_input_dim=0,
    enable_moe=True,
    num_experts=TINY['E'],
    expert_dim=TINY['ED'],
    top_k_experts=TINY['TOPK'],
    moe_dense_hidden_dim=TINY['HF'],
    use_bidirectional_attention='vision',
)

gemma_model = _models.DiffusionGemma_26B_A4B(
    config=tiny_config,
    self_conditioning_config=diffusion_transformer.SelfConditioningConfig(
        features=TINY['D'],
        hidden_dim=TINY['HF'],
    ),
)
wrapped_net = hd_gemma_network.WrappedDiffusionGemmaNetwork(
    gemma_model=gemma_model
)

corruption_process = hd.corruption.CategoricalProcess.uniform_process(
    num_categories=TINY['V'],
    schedule=hd.corruption.RFSchedule(),
)

sft = sft_model.SFTDiffusion(
    x0='batch.canvas',
    prompt='batch.prompt',
    canvas_id='batch.canvas_id',
    canvas_mask='batch.canvas_mask',
    encoder_target='batch.encoder_target',
    encoder_target_mask='batch.encoder_target_mask',
    corruption_process=corruption_process,
    time_sampler=hd.training.time_sampling.UniformTimeSampler(
        span=hd.jax_helpers.SafeSpan(safety_epsilon=1e-4)
    ),
    gemma_network=wrapped_net,
    prompt_len=PROMPT_LEN,
    canvas_size=CANVAS_SIZE,
    num_canvases=NUM_CANVASES,
    stop_gradient_from_denoiser_to_encoder=False,
)

# ---------------------------------------------------------------------------
# Batch built through the REAL data transforms (data.py)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
chunker = data_transforms.CanvasChunker(
    num_canvases=NUM_CANVASES, canvas_size=CANVAS_SIZE,
    eos_token=EOS, pad_token=PAD)
shifter = data_transforms.SequenceTargetShift(pad_token=PAD)

examples = []
# example 0: response fills ~1.5 canvases; example 1: short response, 1 canvas
for resp_len, prompt_valid in [(12, 6), (5, 8)]:
  feats = {
      'prompt': np.concatenate([
          rng.integers(2, TINY['V'], size=prompt_valid),
          np.full(PROMPT_LEN - prompt_valid, PAD)]).astype(np.int32),
      'response': rng.integers(2, TINY['V'], size=resp_len).astype(np.int32),
  }
  feats = chunker.map(feats)
  feats = shifter.map(feats)
  examples.append(feats)

batch = {
    k: np.stack([e[k] for e in examples])
    for k in ('prompt', 'canvas', 'canvas_id', 'canvas_mask',
              'encoder_target', 'encoder_target_mask')
}
batch['canvas'] = batch['canvas'][..., None]  # Rearrange "c -> c 1"
batch = {k: jnp.asarray(v) for k, v in batch.items()}

batch_shapes = {k: f'{v.dtype}{list(v.shape)}' for k, v in batch.items()}
print('batch:', batch_shapes)

# ---------------------------------------------------------------------------
# Module I/O capture
# ---------------------------------------------------------------------------
TRACE = []
PHASE = ['init']


def _shape_of(tree):
  def leaf(x):
    if hasattr(x, 'shape') and hasattr(x, 'dtype'):
      return f'{x.dtype}{list(x.shape)}'
    return repr(x)[:40]
  try:
    return jax.tree.map(leaf, tree)
  except Exception:  # pylint: disable=broad-except
    return str(type(tree).__name__)


def interceptor(next_fun, args, kwargs, context):
  out = next_fun(*args, **kwargs)
  mod = context.module
  TRACE.append({
      'phase': PHASE[0],
      'path': '/'.join(map(str, mod.path)),
      'class': type(mod).__name__,
      'method': context.method_name,
      'inputs': _shape_of({'args': args, 'kwargs': kwargs}),
      'outputs': _shape_of(out),
  })
  return out


model_kwargs = dict(
    x0=batch['canvas'],
    prompt=batch['prompt'],
    canvas_id=batch['canvas_id'],
    canvas_mask=batch['canvas_mask'],
    encoder_target=batch['encoder_target'],
    encoder_target_mask=batch['encoder_target_mask'],
)

# ---------------------------------------------------------------------------
# 1. init + one training step (forward, both losses, backward, optax chain)
# ---------------------------------------------------------------------------
init_rngs = {'params': jax.random.key(0), 'sampling': jax.random.key(1)}
with nn.intercept_methods(interceptor):
  variables = sft.init(init_rngs, **model_kwargs)
params = variables['params']

diffusion_loss_fn = discrete_loss.NoWeightDiscreteLoss(
    use_mask=True, mask_key='target_mask')
encoder_loss_obj = sft_model.EncoderARLoss()

DECODER_LOSS_WEIGHT = 1.0
ENCODER_LOSS_WEIGHT = 1.0


def loss_fn(p, step_rng):
  preds = sft.apply({'params': p}, **model_kwargs,
                    rngs={'sampling': step_rng})
  dl = diffusion_loss_fn(
      preds['output'], preds['target'], preds['noise_info']['time'])  # [B]
  el = encoder_loss_obj.get_values(
      preds['encoder_logits'], preds['encoder_target'],
      preds['encoder_target_mask'])  # [B]
  total = DECODER_LOSS_WEIGHT * jnp.mean(dl) + ENCODER_LOSS_WEIGHT * jnp.mean(el)
  return total, (preds, dl, el)


PHASE[0] = 'train_forward'
with nn.intercept_methods(interceptor):
  (total_loss, (preds, dl, el)), grads = jax.value_and_grad(
      loss_fn, has_aux=True)(params, jax.random.key(2))

preds_shapes = _shape_of(preds)
print('total_loss:', float(total_loss), 'diffusion:', np.asarray(dl),
      'encoder:', np.asarray(el))

# optimizer chain exactly as sft_sudoku_full.py:190-205
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1.5e-4, end_value=1.5e-5,
    warmup_steps=1000, decay_steps=2000)
tx = optax.chain(
    optax.clip_by_global_norm(max_norm=1.0),
    optax.scale_by_factored_rms(),
    optax.add_decayed_weights(weight_decay=1e-4),
    optax.scale_by_learning_rate(schedule),
)
opt_state = tx.init(params)
updates, opt_state = tx.update(grads, opt_state, params)
new_params = optax.apply_updates(params, updates)
# sanity: same treedef and shapes after the update
jax.tree.map(lambda a, b: (_ for _ in ()).throw(
    AssertionError(f'{a.shape} != {b.shape}'))
    if a.shape != b.shape else None, params, new_params)
print('optimizer step OK')

# ---------------------------------------------------------------------------
# 2. AR-diffusion sampling (mirrors ar_eval.py:66-112 + sft_model.py:506-549)
# ---------------------------------------------------------------------------
canvas_sampler = hd.sampling.DiffusionSampler(
    time_schedule=hd.sampling.UniformTimeSchedule(),
    stepper=hd.sampling.DiscreteDDIMStep(
        corruption_process=corruption_process,
        temperature=0.7,
        logits_dtype=jnp.bfloat16,
    ),
    update_conditioning_fn=(
        hd_gemma_ar_state_handler.PropagateSelfConditioningFn()),
    num_steps=NUM_DENOISE_STEPS,
    store_trajectory=False,
)
ar_sampler = sft_model.GemmaKDARSampler(
    state_handler=hd_gemma_ar_state_handler.GemmaARStateHandler(
        gemma_network=wrapped_net,
        gemma_params=params['gemma_network']['gemma_model'],
        pad_token=PAD,
        end_tokens=(EOS,),
    ),
    canvas_sampler=canvas_sampler,
    diffusion_process=corruption_process,
    canvas_length=CANVAS_SIZE,
    max_num_canvases=NUM_CANVASES,
    data_dtype=jnp.int32,
    data_shape=(1,),
)
inference_fn = sft_model.SFTInferenceFn(
    gemma_network=wrapped_net, params=params['gemma_network'])

prompt_tokens = batch['prompt']
prompt_lengths = jnp.sum(prompt_tokens != PAD, axis=-1)

PHASE[0] = 'inference'
with nn.intercept_methods(interceptor):
  final, final_state = ar_sampler(
      diffusion_inference_fn=inference_fn,
      batch_size=B,
      rng=jax.random.key(3),
      conditioning={'prompt_tokens': prompt_tokens,
                    'prompt_lengths': prompt_lengths},
  )

final_state_shapes = _shape_of(final_state)
print('samples:', _shape_of(final))
print('processed_num_canvases:', int(final_state['processed_num_canvases']),
      'processed_denoising_steps:',
      int(final_state['processed_denoising_steps']))

# ---------------------------------------------------------------------------
# 3. Weight inventories
# ---------------------------------------------------------------------------
def inventory(params_tree):
  rows = []
  flat = jax.tree_util.tree_flatten_with_path(params_tree)[0]
  for path, leaf in flat:
    name = '/'.join(
        p.key if hasattr(p, 'key') else str(p) for p in path)
    rows.append({
        'name': name,
        'shape': list(leaf.shape),
        'dtype': str(leaf.dtype),
        'count': int(np.prod(leaf.shape)) if leaf.shape else 1,
        'parent': name.rsplit('/', 1)[0] if '/' in name else '',
    })
  return rows

tiny_inv = inventory(params)
tiny_total = sum(r['count'] for r in tiny_inv)
print('tiny params total:', tiny_total)

# Full 26B_A4B inventory without allocating memory
full_model = _models.DiffusionGemma_26B_A4B()
full_shapes = jax.eval_shape(
    lambda: full_model.init(
        {'params': jax.random.key(0)},
        tokens=jnp.zeros((1, 4), dtype=jnp.int32),
        sc_embeddings=jnp.zeros((1, 4, full_model.config.embed_dim),
                                jnp.float32),
        method=full_model.call_with_self_conditioning,
    )
)
full_inv = inventory(full_shapes['params'])
full_total = sum(r['count'] for r in full_inv)
print('FULL 26B_A4B params total:', full_total, f'({full_total/1e9:.2f}B)')

# ---------------------------------------------------------------------------
# Dump JSON
# ---------------------------------------------------------------------------
with open(os.path.join(OUT_DIR, 'module_io.json'), 'w') as f:
  json.dump(TRACE, f, indent=1, default=str)
with open(os.path.join(OUT_DIR, 'weights_tiny.json'), 'w') as f:
  json.dump({'total': tiny_total, 'params': tiny_inv}, f, indent=1, default=str)
with open(os.path.join(OUT_DIR, 'weights_full_26b.json'), 'w') as f:
  json.dump({'total': full_total, 'params': full_inv}, f, indent=1, default=str)
with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
  json.dump({
      'tiny_dims': TINY,
      'run_dims': {'B': B, 'prompt_len': PROMPT_LEN,
                   'canvas_size': CANVAS_SIZE, 'num_canvases': NUM_CANVASES,
                   'total_canvas_len': TOTAL_CANVAS_LEN,
                   'full_seq_len': FULL_SEQ_LEN,
                   'num_denoise_steps': NUM_DENOISE_STEPS},
      'batch': batch_shapes,
      'preds': preds_shapes,
      'losses': {'total': float(total_loss),
                 'diffusion_per_example': np.asarray(dl).tolist(),
                 'encoder_per_example': np.asarray(el).tolist()},
      'samples_shape': _shape_of(final),
      'final_state': final_state_shapes,
      'tiny_param_total': tiny_total,
      'full_param_total': full_total,
  }, f, indent=1, default=str)
print('wrote JSON to', OUT_DIR)
