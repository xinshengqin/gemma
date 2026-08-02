# DiffusionGemma batch-1 inference benchmark — plan

Approved 2026-08-02. Vocabulary: [CONTEXT.md](./CONTEXT.md). Full decision log with
rejected alternatives and trade-offs: [decision.md](./decision.md).

## Goal

Measure batch-1 inference latency/throughput of `DIFFUSIONGEMMA_26B_A4B_IT` through the
native JAX `gemma.diffusion.Sampler`, reported in the metric vocabulary of Fast-dDrive
(arXiv 2605.23163), with the infra validated by first reproducing vLLM's published H100
number on the same box. Learnings that transfer to a future benchmark of the
hackable_diffusion/vision stack are captured in `LEARNINGS.md`.

## References

| Reference | Setup | Number |
|---|---|---|
| vLLM blog (2026-06-10) | 1× H100, FP8 checkpoint, batch 1, `vllm bench serve` | 1,008 generation tok/s |
| vLLM repro gist (LucasWilkinson) | random 1024 in / 1024 out, ignore-EOS, 100 prompts, concurrency 1; canvas 256, 16 denoising steps | protocol source |
| Fast-dDrive (arXiv 2605.23163) | 1× H100, batch 1, ~280-token structured outputs | latency ms/sample, TPS, Tok/Step; AR baseline 7855 ms / 51.6 TPS → best 665 ms / 608.5 TPS |

Every public H100 DiffusionGemma number is an optimized serving stack at FP8; there is no
published JAX-native or bf16 reference. Our absolute numbers are expected to land below
1,008 for stack-independent reasons (bf16 weights, double-apply self-conditioning,
research-grade JAX).

## Configuration (fixed across all timed runs)

Batch 1 · canvas 256 · 16 denoising steps · `NoEarlyStop()` · EOS ignored · bf16 ·
fixed shapes only. Two workloads (calibration, paper-shaped) and four metrics as defined
in [CONTEXT.md](./CONTEXT.md).

## Pass criteria

- vLLM FP8 repro within ~10% of 1,008 generation tok/s. A miss means debug the box; do
  not proceed to JAX runs on unvalidated infra.
- Sanity-run output is coherent text.

## Execution order

1. Local, before renting: build `bench_native.py`, `bench_vllm.sh`, `setup_vast.sh`;
   smoke-test the harness on CPU with the repo's tiny test config
   (`gemma/diffusion/_sampler_test.py` `_SMALL_CONFIG`); verify checkpoint accessibility
   from outside Google infra (`gs://gemma-data/...`; fallback HF/Kaggle copy).
2. Rent H100 → `setup_vast.sh` → vLLM FP8 repro → check pass criterion.
3. Sanity run, then JAX calibration and paper-shaped runs.
4. Pull results JSONs + sanity output, destroy the instance (destroy, not stop), write
   results table and `LEARNINGS.md`, commit locally.

## Layout

```
perf/
├── docs/
│   ├── PLAN.md          # this file
│   ├── decision.md      # decision log: alternatives rejected and trade-offs
│   ├── CONTEXT.md       # metric/workload glossary
│   └── LEARNINGS.md     # written after the runs; feeds the future HD/vision benchmark
├── bench_native.py      # JAX native-sampler benchmark (both workloads → results/*.json)
├── bench_vllm.sh        # gist-derived vLLM FP8 repro with the same provenance capture
├── setup_vast.sh        # rental bootstrap: CUDA/jax[cuda13]/vLLM, checkpoints, XLA/NCCL env
└── results/             # committed JSONs, one per (stack × workload), + sanity output text
```

## Known risks

- `gs://gemma-data` may require auth from the rental box → pre-verified in step 1;
  fallback is the HF/Kaggle checkpoint copy.
- Native sampler may lack an ignore-EOS switch → minimal local patch, documented in
  LEARNINGS.md.
- 1024-token prompts vs sampler pad buckets (256/512/1024) and `cache_length=4096` →
  checked during the CPU smoke test.
