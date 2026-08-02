# DiffusionGemma batch-1 inference benchmark — plan

Approved 2026-08-02. Vocabulary: [context.md](./context.md). Full decision log with
rejected alternatives and trade-offs: [decision.md](./decision.md). Results land in
[report.md](./report.md); mistakes and recovered failures along the way land in
[learnings.md](./learnings.md).

## Goal

Measure batch-1 inference latency/throughput of `DIFFUSIONGEMMA_26B_A4B_IT` through the
native JAX `gemma.diffusion.Sampler`, reported in the metric vocabulary of Fast-dDrive
(arXiv 2605.23163), with the infra validated by first reproducing vLLM's published H100
number on the same box. Findings that transfer to a future benchmark of the
hackable_diffusion/vision stack are captured in `report.md`.

## References

| Reference | Setup | Number |
|---|---|---|
| vLLM blog (2026-06-10) | 1× H100, FP8 checkpoint, batch 1, `vllm bench serve` | 1,008 generation tok/s |
| vLLM repro gist (LucasWilkinson) | random 1024 in / 1024 out, ignore-EOS, 100 prompts, concurrency 1; canvas 256, 16 denoising steps | protocol source |
| Fast-dDrive (arXiv 2605.23163) | 1× H100, batch 1, ~280-token structured outputs | latency ms/sample, TPS, "Tok/Step" (our Tok/Forward); AR baseline 7855 ms / 51.6 TPS → best 665 ms / 608.5 TPS |

Every public H100 DiffusionGemma number is an optimized serving stack at FP8; there is no
published JAX-native or bf16 reference. Our absolute numbers are expected to land below
1,008 for stack-independent reasons (bf16 weights, double-apply self-conditioning,
research-grade JAX).

## Configuration (fixed across all timed runs)

Batch 1 · canvas 256 · 16 denoising steps · `NoEarlyStop()` · EOS ignored · bf16 ·
fixed shapes only. Two workloads (calibration, paper-shaped) and four metrics as defined
in [context.md](./context.md).

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
4. Pull results JSONs + sanity output, destroy the instance (destroy, not stop), fill in
   `report.md` (and `learnings.md` if anything went wrong along the way), commit locally.

## Milestones and intermediate reporting

Requirement: at each milestone below, update [status.md](./status.md) and commit (pushed
to the PR branch) **before** starting the next milestone, so that if work is interrupted
or diverges, anyone can resume from what has already been achieved without redoing it.

| # | Milestone | Definition of done |
|---|---|---|
| M0 | Planning docs approved | this doc set merged-reviewable on the PR |
| M1 | Harness ready | scripts built; CPU smoke test passing; checkpoint access from outside Google infra verified |
| M2 | Box bootstrapped | H100 rented; `setup_vast.sh` completed; checkpoints downloaded |
| M3 | Infra calibrated | vLLM FP8 repro done; gate verdict recorded in report.md |
| M4 | JAX runs complete | sanity run + calibration + paper-shaped results JSONs pulled and committed |
| M5 | Wrapped up | instance destroyed; report.md finalized; learnings.md updated if anything went wrong |

Each status.md update records: what was achieved (with artifact paths), any deviation
from plan/decisions, open blockers, and the exact next step (command-level). Results
JSONs and partial report.md fills are committed as they are produced at each milestone —
never held back for the end. A live rental (M2–M4) is the interruption-sensitive window:
status.md must always contain enough to re-rent a box and resume without repeating
finished stages.

## Layout

```
perf/
├── docs/
│   ├── plan.md          # this file
│   ├── decision.md      # decision log: alternatives rejected and trade-offs
│   ├── context.md       # metric/workload glossary
│   ├── report.md        # final results (prefilled with placeholders until runs complete)
│   ├── status.md        # milestone log: rolling intermediate reports for resumability
│   └── learnings.md     # major mistakes made / failures recovered during the project
├── bench_native.py      # JAX native-sampler benchmark (both workloads → results/*.json)
├── bench_vllm.sh        # gist-derived vLLM FP8 repro with the same provenance capture
├── setup_vast.sh        # rental bootstrap: CUDA/jax[cuda13]/vLLM, checkpoints, XLA/NCCL env
└── results/             # committed JSONs, one per (stack × workload), + sanity output text
```

## Known risks

- `gs://gemma-data` may require auth from the rental box → pre-verified in step 1;
  fallback is the HF/Kaggle checkpoint copy.
- Native sampler may lack an ignore-EOS switch → minimal local patch, documented in
  report.md.
- 1024-token prompts vs sampler pad buckets (256/512/1024) and `cache_length=4096` →
  checked during the CPU smoke test.
