"""Batch-1 inference benchmark for the native JAX diffusion sampler.

Modes (plan: perf/docs/plan.md, vocabulary: perf/docs/context.md):
  smoke        CPU harness check on a tiny random-init model (no checkpoint).
               Empirically verifies the Tok/Forward accounting by counting
               model.apply calls, then exercises the jitted timed path + JSON.
  sanity       Untimed generations on 5 real prompts, outputs saved for
               eyeballing (perf/results/sanity_outputs.md).
  calibration  Random-token prompts, 1024 in / 1024 out forced, timed.
  paper        Fixed real prompt, 280-token forced output (512 emitted), timed.

All timed runs: batch 1, canvas 256, 16 denoising steps, NoEarlyStop, EOS
ignored, checkpoint dtype (bf16), fixed shapes. EOS is ignored via
end_tokens=(): with an empty tuple both the stop-token truncation in
DiffusionSampler._sample_step and the post-loop _mask_tokens_after_end_tokens
are no-ops, so no repo patch is needed.

Timed requests bypass the text pipeline (gm.text.Sampler is str-only) and call
_prefill.prefill + DiffusionSampler.sample directly with token IDs. BOS is
prepended manually; random prompt IDs are drawn from [3, vocab) because token 0
is PAD and corrupts the input masks.
"""

import argparse
import dataclasses
import importlib.metadata
import json
import math
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gemma import gm
from gemma import diffusion
from gemma.diffusion import _early_stopping
from gemma.diffusion import _sampler as _dsampler
from gemma.diffusion import _transformer as _dtransformer
from gemma.gm.nn.gemma4 import _config as _g4config
from gemma.gm.nn.gemma4 import _modules as _g4modules
from gemma.gm.text import _prefill
from gemma.gm.text import _sampling
from gemma.gm.utils import _types

CANVAS_LENGTH = 256
DENOISING_STEPS = 16
CACHE_LENGTH = 4096
PAD_BUCKETS = (256, 512, 1024)

PAPER_PROMPT = (
    'Explain how a rechargeable lithium-ion battery works, covering the roles'
    ' of the anode, cathode, electrolyte, and separator during both charge and'
    ' discharge.'
)

SANITY_PROMPTS = (
    'What causes the seasons on Earth? Answer in one short paragraph.',
    'Write a four-line poem about a lighthouse keeper.',
    'Write a Python function that returns the n-th Fibonacci number'
    ' iteratively.',
    'A train travels 240 km in 3 hours, then 180 km in 2 hours. What is its'
    ' average speed for the whole trip? Show your reasoning.',
    'Summarize in two sentences: The Great Barrier Reef is the world\'s'
    ' largest coral reef system, composed of over 2,900 individual reefs and'
    ' 900 islands stretching for over 2,300 kilometres. It supports a wide'
    ' diversity of life and was selected as a World Heritage Site in 1981.',
)


def _make_loop(
    model,
    *,
    vocab_size,
    canvas_length,
    denoising_steps,
    cache_length,
    special_tokens,
    sliding_window_size,
):
  """DiffusionSampler configured for forced-length, ignore-EOS benchmarking."""
  return _dsampler.DiffusionSampler(
      model=model,
      end_tokens=(),
      forbidden_tokens=None,
      sampling=_sampling.Greedy(),  # inert: DiffusionSampler overrides _sample_step
      cache_length=cache_length,
      special_tokens=special_tokens,
      diffusion_process=_dsampler.DiffusionProcess(),
      logit_shaper=_dsampler.AnnealingTemperatureShaperConfig().make(),
      sample_from_predictions=_dsampler.SampleFromPredictions(
          entropy_bound=0.1, text_vocab_size=vocab_size
      ),
      canvas_length=canvas_length,
      max_denoising_steps=denoising_steps,
      text_vocab_size=vocab_size,
      sliding_window_size=sliding_window_size,
      early_stop_fn=_early_stopping.NoEarlyStop(),
  )


def _run_request(
    *, model, params, loop, tokens, max_new_tokens, max_out_length,
    cache_length, pad_length, rng,
):
  """One request. Returns (final state, prefill seconds, decode seconds)."""
  t0 = time.perf_counter()
  inp = _types.Input(
      text=jnp.asarray(tokens, dtype=jnp.int32)[None, :],
      images=None,
      config=model.config.input_config,
  )
  init_state = _prefill.prefill(
      model=model,
      params=params,
      input=inp,
      last_state=None,
      cache_length=cache_length,
      max_out_length=max_out_length,
      pad_length=pad_length,
      rng=rng,
      sharding=None,
  )
  jax.block_until_ready(init_state)
  t1 = time.perf_counter()
  state = loop.sample(
      params=params,
      init_state=init_state,
      max_new_tokens=jnp.asarray(max_new_tokens),
      stream=False,
  )
  jax.block_until_ready(state)
  t2 = time.perf_counter()
  return state, t1 - t0, t2 - t1


def _forward_accounting(*, max_new_tokens, canvas_length, denoising_steps,
                        delivered_tokens):
  """Deterministic forward-pass accounting (context.md: Tok/Forward)."""
  canvases = math.ceil(max_new_tokens / canvas_length)
  denoiser_forwards = canvases * denoising_steps
  cache_append_forwards = canvases
  decode_forwards = denoiser_forwards + cache_append_forwards
  emitted_tokens = canvases * canvas_length
  return {
      'canvases': canvases,
      'denoiser_forwards': denoiser_forwards,
      'cache_append_forwards': cache_append_forwards,
      'decode_forwards': decode_forwards,
      'prefill_forwards': 1,
      'encode_logits_applies': denoiser_forwards,  # embedder ops, not forwards
      'emitted_tokens': emitted_tokens,
      'delivered_tokens': delivered_tokens,
      'tok_per_forward_delivered': delivered_tokens / decode_forwards,
      'tok_per_forward_emitted': emitted_tokens / decode_forwards,
  }


def _stats(xs):
  a = np.asarray(xs, dtype=np.float64)
  return {
      'median': float(np.median(a)),
      'mean': float(np.mean(a)),
      'p95': float(np.percentile(a, 95)),
  }


def _nvidia_smi():
  try:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=10, check=True,
    ).stdout.strip()
    banner = subprocess.run(
        ['nvidia-smi'], capture_output=True, text=True, timeout=10, check=True,
    ).stdout
    cuda = None
    for line in banner.splitlines():
      if 'CUDA Version' in line:
        cuda = line.split('CUDA Version:')[1].strip(' |').strip()
        break
    return {'gpu': out, 'cuda_version': cuda}
  except (OSError, subprocess.SubprocessError):
    return None


def _git_commit():
  try:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        capture_output=True, text=True, timeout=10, check=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()
  except (OSError, subprocess.SubprocessError):
    return None


def _provenance(*, checkpoint, tokenizer_path, param_dtype):
  return {
      'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
      'hostname': socket.gethostname(),
      'python': sys.version.split()[0],
      'jax': jax.__version__,
      'jaxlib': importlib.metadata.version('jaxlib'),
      'backend': jax.default_backend(),
      'devices': [d.device_kind for d in jax.devices()],
      'nvidia_smi': _nvidia_smi(),
      'xla_flags': os.environ.get('XLA_FLAGS'),
      'xla_python_client_preallocate': os.environ.get(
          'XLA_PYTHON_CLIENT_PREALLOCATE'
      ),
      'git_commit': _git_commit(),
      'checkpoint': str(checkpoint),
      'tokenizer': str(tokenizer_path),
      'param_dtype': param_dtype,
  }


def _timed_workload(
    *, workload, model, params, loop, prompts, max_new_tokens,
    delivered_tokens, max_out_length, cache_length, pad_length,
    num_prompts, warmup, seed, provenance, out_path, sampler_config_extra,
):
  """Compile + warmup + timed requests over `prompts` (one per request).

  prompts must all share one shape so everything compiles exactly once
  (decision.md section 9). Request i uses prompts[i]; there must be
  1 + warmup + num_prompts of them.
  """
  total = 1 + warmup + num_prompts
  assert len(prompts) == total, (len(prompts), total)
  assert all(len(p) == len(prompts[0]) for p in prompts)

  def request(i):
    return _run_request(
        model=model, params=params, loop=loop, tokens=prompts[i],
        max_new_tokens=max_new_tokens, max_out_length=max_out_length,
        cache_length=cache_length, pad_length=pad_length,
        rng=jax.random.key(seed * 1_000_000 + i),
    )

  print(f'[{workload}] compile + first sample...', flush=True)
  t0 = time.perf_counter()
  request(0)
  compile_plus_first_sample_s = time.perf_counter() - t0
  print(f'[{workload}] compile + first sample: '
        f'{compile_plus_first_sample_s:.1f}s', flush=True)

  for i in range(1, 1 + warmup):
    request(i)
  print(f'[{workload}] {warmup} warmup samples done', flush=True)

  prefill_s, decode_s, latency_s = [], [], []
  last_state = None
  for i in range(1 + warmup, total):
    last_state, p, d = request(i)
    prefill_s.append(p)
    decode_s.append(d)
    latency_s.append(p + d)
    if (i - warmup) % 10 == 0:
      print(f'[{workload}] timed {i - warmup}/{num_prompts}', flush=True)

  # Post-hoc invariants on the last request.
  assert int(last_state.step) == math.ceil(
      max_new_tokens / loop.canvas_length) * loop.canvas_length
  assert not bool(jnp.any(last_state.done))  # EOS truly ignored

  accounting = _forward_accounting(
      max_new_tokens=max_new_tokens,
      canvas_length=loop.canvas_length,
      denoising_steps=loop.max_denoising_steps,
      delivered_tokens=delivered_tokens,
  )
  result = {
      'stack': 'jax_native',
      'workload': workload,
      'config': {
          'batch_size': 1,
          'canvas_length': loop.canvas_length,
          'max_denoising_steps': loop.max_denoising_steps,
          'early_stop': 'NoEarlyStop',
          'eos_ignored': True,
          'entropy_bound': 0.1,
          'logit_shaper': dataclasses.asdict(loop.logit_shaper.config),
          'cache_length': cache_length,
          'max_out_length': max_out_length,
          'pad_length': pad_length,
          'input_tokens': len(prompts[0]),
          'input_includes_bos': True,
          'max_new_tokens': max_new_tokens,
          'num_prompts': num_prompts,
          'warmup': warmup,
          'seed': seed,
          **sampler_config_extra,
      },
      'provenance': provenance,
      'compile': {'compile_plus_first_sample_s': compile_plus_first_sample_s},
      'forward_accounting': accounting,
      'metrics': {
          'latency_ms': _stats([s * 1e3 for s in latency_s]),
          'generation_tok_s': _stats(
              [delivered_tokens / d for d in decode_s]
          ),
          'end_to_end_tok_s': _stats(
              [delivered_tokens / l for l in latency_s]
          ),
          'prefill_ms': _stats([s * 1e3 for s in prefill_s]),
          'tok_per_forward': accounting['tok_per_forward_delivered'],
      },
      'per_request': {
          'prefill_s': prefill_s,
          'decode_s': decode_s,
          'latency_s': latency_s,
      },
  }
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(result, indent=2) + '\n')
  print(f'[{workload}] median latency '
        f"{result['metrics']['latency_ms']['median']:.0f} ms, "
        f'median generation '
        f"{result['metrics']['generation_tok_s']['median']:.1f} tok/s -> "
        f'{out_path}', flush=True)
  return result


def _load_real_model(checkpoint, tokenizer_path):
  model = diffusion.DiffusionGemma_26B_A4B()
  params = gm.ckpts.load_params(checkpoint, restore_concurrent_gb=16)
  tokenizer = (
      gm.text.Gemma4Tokenizer(path=tokenizer_path)
      if tokenizer_path
      else gm.text.Gemma4Tokenizer()
  )
  return model, params, tokenizer


def run_timed(args):
  model, params, tokenizer = _load_real_model(args.checkpoint, args.tokenizer)
  vocab = tokenizer.vocab_size
  st = tokenizer.special_tokens
  loop = _make_loop(
      model,
      vocab_size=vocab,
      canvas_length=CANVAS_LENGTH,
      denoising_steps=DENOISING_STEPS,
      cache_length=CACHE_LENGTH,
      special_tokens=st,
      sliding_window_size=getattr(model.config, 'sliding_window_size', None),
  )
  total = 1 + args.warmup + args.num_prompts
  if args.mode == 'calibration':
    # vLLM-gist shape: 1024 input tokens (BOS + 1023 random), 1024 forced out.
    # IDs from [3, vocab): 0 is PAD (corrupts masks), 1/2 are EOS/BOS.
    gen = np.random.default_rng(args.seed)
    prompts = [
        np.concatenate((
            [st.BOS],
            gen.integers(3, vocab, size=1023, dtype=np.int32),
        ))
        for _ in range(total)
    ]
    max_new_tokens, delivered = 1024, 1024
  else:  # paper
    ids = np.asarray(tokenizer.encode(PAPER_PROMPT, add_bos=True))
    assert len(ids) <= PAD_BUCKETS[0], len(ids)
    prompts = [ids] * total
    max_new_tokens, delivered = 280, 280

  emitted = math.ceil(max_new_tokens / CANVAS_LENGTH) * CANVAS_LENGTH
  provenance = _provenance(
      checkpoint=args.checkpoint,
      tokenizer_path=args.tokenizer or gm.text.Gemma4Tokenizer.path,
      param_dtype=str(jax.tree.leaves(params)[0].dtype),
  )
  _timed_workload(
      workload=args.mode,
      model=model, params=params, loop=loop, prompts=prompts,
      max_new_tokens=max_new_tokens, delivered_tokens=delivered,
      max_out_length=emitted, cache_length=CACHE_LENGTH,
      pad_length=PAD_BUCKETS, num_prompts=args.num_prompts,
      warmup=args.warmup, seed=args.seed, provenance=provenance,
      out_path=args.outdir / f'jax_native_{args.mode}.json',
      sampler_config_extra=(
          {} if args.mode == 'calibration' else {'paper_prompt': PAPER_PROMPT}
      ),
  )


def run_sanity(args):
  model, params, tokenizer = _load_real_model(args.checkpoint, args.tokenizer)
  chat = diffusion.ChatSampler(
      model=model,
      params=params,
      tokenizer=tokenizer,
      canvas_length=CANVAS_LENGTH,
      max_denoising_steps=DENOISING_STEPS,
      early_stop_fn=_early_stopping.NoEarlyStop(),
      cache_length=CACHE_LENGTH,
      max_out_length=1024,
      pad_length=PAD_BUCKETS,
  )
  lines = [
      '# Sanity run outputs',
      '',
      f'Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}'
      f' with canvas {CANVAS_LENGTH}, {DENOISING_STEPS} denoising steps,'
      ' NoEarlyStop, EOS respected (untimed; see perf/docs/context.md).',
      '',
  ]
  for i, prompt in enumerate(SANITY_PROMPTS, 1):
    print(f'[sanity] prompt {i}/5...', flush=True)
    out = chat.chat(prompt, rng=i)
    lines += [f'## Prompt {i}', '', prompt, '', '### Output', '', out, '']
  args.outdir.mkdir(parents=True, exist_ok=True)
  path = args.outdir / 'sanity_outputs.md'
  path.write_text('\n'.join(lines))
  print(f'[sanity] outputs -> {path}', flush=True)


def run_smoke(args):
  """CPU-only harness check on a tiny random-init model."""
  canvas_length, denoising_steps, vocab, cache_length = 8, 4, 32, 128
  cfg = _g4config.TransformerConfig(
      num_embed=vocab,
      embed_dim=8,
      num_heads=2,
      num_kv_heads=1,
      head_dim=4,
      hidden_dim=16,
      # Tuple, not the test's verbatim list: the sampler loop jit hashes the
      # model as a static arg, and the tests only run under disable_jit.
      attention_types=(_g4modules.AttentionType.GLOBAL,),
      kv_cache_sharing_config=None,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      final_logit_softcap=None,
      global_rope_proportion=1.0,
  )
  model = diffusion.DiffusionGemma_26B_A4B(
      config=cfg,
      self_conditioning_config=_dtransformer.SelfConditioningConfig(
          features=cfg.embed_dim, hidden_dim=cfg.hidden_dim
      ),
  )
  params = model.init(
      rngs=jax.random.PRNGKey(0),
      tokens=jnp.ones((1, canvas_length), dtype=jnp.int32),
      sc_embeddings=jnp.ones(
          (1, canvas_length, cfg.embed_dim), dtype=jnp.bfloat16
      ),
      attention_mask=jnp.ones(
          (1, canvas_length, canvas_length), dtype=jnp.bool_
      ),
      method=model.call_with_self_conditioning,
  )['params']
  loop = _make_loop(
      model,
      vocab_size=vocab,
      canvas_length=canvas_length,
      denoising_steps=denoising_steps,
      cache_length=cache_length,
      special_tokens=None,
      sliding_window_size=None,
  )

  # 1. Pad-bucket behavior for the 1024-token calibration prompts (plan.md
  # "Known risks"): 1024 hits the largest bucket exactly; over-bucket lengths
  # fall back to raw length (fixed shape as long as all prompts match).
  assert _prefill._pad_to_bucket(1024, PAD_BUCKETS) == 1024
  assert _prefill._pad_to_bucket(1025, PAD_BUCKETS) == 1025

  # 2. Empirical forward-count check of the Tok/Forward denominator
  # (decision.md section 6): count model.apply calls eagerly.
  counts = {'denoiser': 0, 'plain': 0, 'embedder': 0}
  orig_apply = model.apply

  def counting_apply(*a, **kw):
    method = kw.get('method')
    if method is _dtransformer.DiffusionMixin.call_with_self_conditioning:
      counts['denoiser'] += 1
    elif method is None:
      counts['plain'] += 1  # prefill or cache-append forward
    else:
      counts['embedder'] += 1  # encode_logits self-conditioning apply
    return orig_apply(*a, **kw)

  gen = np.random.default_rng(args.seed)
  prompt = np.concatenate(([2], gen.integers(3, vocab, size=15, dtype=np.int32)))
  max_new_tokens = 16  # 2 canvases of 8
  object.__setattr__(model, 'apply', counting_apply)
  try:
    with jax.disable_jit():
      state, _, _ = _run_request(
          model=model, params=params, loop=loop, tokens=prompt,
          max_new_tokens=max_new_tokens, max_out_length=max_new_tokens,
          cache_length=cache_length, pad_length=None,
          rng=jax.random.key(args.seed),
      )
  finally:
    object.__delattr__(model, 'apply')

  expected = _forward_accounting(
      max_new_tokens=max_new_tokens,
      canvas_length=canvas_length,
      denoising_steps=denoising_steps,
      delivered_tokens=max_new_tokens,
  )
  assert counts['denoiser'] == expected['denoiser_forwards'], counts
  assert counts['plain'] == expected['prefill_forwards'] + expected[
      'cache_append_forwards'], counts
  assert counts['embedder'] == expected['encode_logits_applies'], counts
  assert int(state.step) == expected['emitted_tokens']
  assert not bool(jnp.any(state.done))
  print(f'[smoke] forward counts verified: {counts} == '
        f"{{'denoiser': {expected['denoiser_forwards']}, 'plain': "
        f"{expected['prefill_forwards'] + expected['cache_append_forwards']}, "
        f"'embedder': {expected['encode_logits_applies']}}}", flush=True)

  # 3. Jitted timed path end to end, including stats + JSON output.
  total = 1 + 1 + 2  # compile + 1 warmup + 2 timed
  prompts = [
      np.concatenate(([2], gen.integers(3, vocab, size=15, dtype=np.int32)))
      for _ in range(total)
  ]
  result = _timed_workload(
      workload='smoke',
      model=model, params=params, loop=loop, prompts=prompts,
      max_new_tokens=max_new_tokens, delivered_tokens=max_new_tokens,
      max_out_length=max_new_tokens, cache_length=cache_length,
      pad_length=None, num_prompts=2, warmup=1, seed=args.seed,
      provenance=_provenance(
          checkpoint='none (random init)', tokenizer_path='none (raw ids)',
          param_dtype=str(jax.tree.leaves(params)[0].dtype),
      ),
      out_path=args.outdir / 'smoke.json',
      sampler_config_extra={'smoke_tiny_model': True},
  )
  assert result['forward_accounting']['tok_per_forward_delivered'] == (
      16 / (2 * (denoising_steps + 1))
  )
  print('[smoke] PASS', flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      'mode', choices=['smoke', 'sanity', 'calibration', 'paper']
  )
  parser.add_argument(
      '--checkpoint',
      default=str(diffusion.CheckpointPath.DIFFUSIONGEMMA_26B_A4B_IT),
      help='orbax checkpoint dir (gs:// or local)',
  )
  parser.add_argument(
      '--tokenizer', default=None,
      help='tokenizer .model path (default: gs:// Gemma4 tokenizer)',
  )
  parser.add_argument(
      '--outdir', type=Path, default=Path(__file__).parent / 'results'
  )
  parser.add_argument('--num-prompts', type=int, default=100)
  parser.add_argument('--warmup', type=int, default=3)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()

  if args.mode == 'smoke':
    run_smoke(args)
  elif args.mode == 'sanity':
    run_sanity(args)
  else:
    run_timed(args)


if __name__ == '__main__':
  main()
