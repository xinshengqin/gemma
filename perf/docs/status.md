# Status — milestone log

Rolling intermediate reports, one entry per milestone (defined in
[plan.md](./plan.md#milestones-and-intermediate-reporting)). Updated and committed at
every milestone boundary so anyone can resume the work from the last entry without
redoing finished stages. Newest entry first.

## Resume procedure

1. Read the newest entry below: it states what is done, artifact paths, deviations, and
   the exact next step.
2. Never redo a milestone marked done — its artifacts are committed.
3. If a rental was lost mid-window (M2–M4), re-rent per plan.md ("The rental" spec in
   decision.md §8), rerun `setup_vast.sh`, and continue from the newest entry's next
   step; committed results JSONs in `perf/results/` mark which runs are already banked.

## Milestones

| # | Milestone | Status |
|---|---|---|
| M0 | Planning docs approved | in review (PR #5) |
| M1 | Harness ready | done |
| M2 | Box bootstrapped | done |
| M3 | Infra calibrated | not started |
| M4 | JAX runs complete | not started |
| M5 | Wrapped up | not started |

## Entries

### 2026-08-02 — M2: box bootstrapped

- **Achieved**: vast.ai instance **46665143** (user pre-authorized full M2–M5
  execution this session). 1× H100 SXM 80GB HBM3, driver 595.71.05 (CUDA 13.2),
  $2.3507/hr on-demand, 200 GB disk, Czechia, ~8.7 Gbps down; ssh
  `root@93.91.156.108:43518`. Image `vllm/vllm-openai:gemma` (vllm
  0.22.1rc1.dev357+g74b5964f0, python 3.12.13). `setup_vast.sh` exit 0 on first
  run: repo cloned to `/workspace/gemma` (this branch), JAX venv at
  `/workspace/venv-jax` (jax/jaxlib 0.11.0 + jax-cuda13 0.11.0, flax 0.12.8,
  orbax-checkpoint 0.12.1; `jax.default_backend()=='gpu'`), checkpoint (32
  objects) at `/workspace/ckpt/diffusiongemma-26B-A4B-it`, tokenizer alongside,
  env in `/workspace/env.sh`, provenance in `/workspace/provenance_setup.txt`.
- **Deviations**: image has no `/workspace` by default — created before launch
  (setup script unaffected; WORKDIR was already parameterized).
- **Blockers**: none. Billing live — destroy (not stop) 46665143 at wrap-up.
- **Next step**: M3 — on the box: `bash /workspace/gemma/perf/bench_vllm.sh`;
  gate = median generation tok/s within ~10% of 1,008.

### 2026-08-02 — M1: harness ready

- **Achieved**: `perf/bench_native.py` (modes: smoke / sanity / calibration /
  paper), `perf/bench_vllm.sh`, `perf/setup_vast.sh`, on branch
  `worktree-perf-exec`.
  - CPU smoke test passing (venv `/home/xqin/projects/gemma/.venv`): pad-bucket
    asserts (1024→bucket 1024; 1025→raw length, no error), empirical
    forward-count verification by counting `model.apply` calls eagerly
    (denoiser = canvases × steps, plain = 1 prefill + 1 cache-append per
    canvas, embedder-only `encode_logits` = denoiser count — confirms the
    Tok/Forward denominator of decision.md §6), jitted timed path + stats +
    results-JSON plumbing.
  - Ignore-EOS needs **no repo patch** (plan.md risk retired): `end_tokens=()`
    makes both the canvas stop-token truncation and the post-loop
    `_mask_tokens_after_end_tokens` no-ops.
  - Checkpoint access from outside Google infra **verified**: `gs://gemma-data`
    is anonymously readable (checkpoint 40.4 GB / 32 objects, orbax ocdbt;
    Gemma4 tokenizer public too) — `setup_vast.sh` downloads via plain HTTPS,
    no auth. HF fallback `google/diffusiongemma-26B-A4B-it` is safetensors-only
    (wrong format for the JAX sampler; emergency only). vLLM FP8 repro
    checkpoint: `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic` (public,
    not gated).
  - Gemma4 tokenizer verified locally: vocab 262144, paper-shaped prompt =
    30 tokens (fits pad bucket 256).
- **Deviations**: none from plan. Pre-push adversarial review confirmed and
  fixed 3 findings: curl HTTP-error bodies could pass as checkpoint shards and
  the download sentinel was itself a downloaded object (both `setup_vast.sh`);
  vLLM results JSON lacked the `--hf-overrides` sampler knobs and CUDA version
  required by decision.md §9 (`bench_vllm.sh`).
- **Blockers**: PR #5 (M0 docs) still awaiting user review. M2 spends money —
  wait for user go.
- **Next step**: M2 — rent 1× H100 SXM 80GB on-demand (decision.md §8,
  ≥200 GB disk, image `vllm/vllm-openai:gemma`) via `vastai` CLI; on the box:
  `bash perf/setup_vast.sh`, then M3: `bash perf/bench_vllm.sh` (gate vs
  1,008 tok/s).

### 2026-08-02 — M0: planning docs

- **Achieved**: full doc set under `perf/docs/` (plan, decision log, glossary, report
  skeleton, learnings charter, this file) on branch `worktree-perf`, PR #5.
- **Deviations**: none — planning phase.
- **Blockers**: awaiting PR review/approval.
- **Next step**: M1 — build `perf/bench_native.py`, `perf/bench_vllm.sh`,
  `perf/setup_vast.sh`; CPU smoke test via
  `gemma/diffusion/_sampler_test.py::_SMALL_CONFIG`; verify `gs://gemma-data`
  checkpoint accessibility from a non-Google box (fallback: HF/Kaggle copy).
