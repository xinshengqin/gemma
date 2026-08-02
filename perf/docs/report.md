# DiffusionGemma batch-1 inference benchmark — report

**Status: DRAFT — runs not yet executed; `TBD` marks values pending experiment results.**

Plan: [plan.md](./plan.md) · decisions: [decision.md](./decision.md) · vocabulary:
[context.md](./context.md). All timed runs: batch 1, canvas 256, 16 denoising steps,
`NoEarlyStop()`, EOS ignored, fixed shapes, single H100 SXM 80GB.

## Summary

> TBD after runs. Template: "The native JAX sampler at bf16 delivers **TBD** generation
> tok/s (median, calibration workload) vs vLLM's 1,008 at FP8 on the same box — a **TBD×**
> gap attributable to stack + precision. Infra was validated by reproducing vLLM's
> published number to within **TBD%**. On the Fast-dDrive-shaped workload (280-token
> outputs) it delivers **TBD** ms/sample."

## Setup

| Item | Value |
|---|---|
| GPU | 1× H100 SXM 80GB (vast.ai instance TBD, hourly rate TBD) |
| Driver / CUDA | TBD / TBD |
| JAX / jaxlib | TBD |
| vLLM | TBD |
| XLA / NCCL env | TBD (from `setup_vast.sh`) |
| JAX checkpoint | `DIFFUSIONGEMMA_26B_A4B_IT` (`gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it`), bf16 |
| vLLM checkpoint | FP8 (per vLLM repro gist) |
| Total rental cost | TBD |

Full provenance for every run: `perf/results/*.json` (TBD).

## Infra calibration (gate)

Pass criterion: within ~10% of vLLM's published 1,008 generation tok/s (H100, FP8, batch 1).

| Metric | Published | Measured | Ratio | Verdict |
|---|---|---|---|---|
| Generation tok/s (median) | 1,008 | TBD | TBD | TBD |

## Sanity run

Untimed generation on a real prompt to confirm the checkpoint loaded correctly.
Output saved at `perf/results/sanity_output.txt` (TBD). Verdict: TBD.

## Results

All values over 100 timed samples after JIT compile + 3 warmup samples.

| Stack | Workload | Latency ms/sample (median) | Generation tok/s (median) | End-to-end tok/s | Tok/Step |
|---|---|---|---|---|---|
| vLLM FP8 | calibration (1024 in / 1024 out) | TBD | TBD | TBD | 16.0 nominal¹ |
| JAX native bf16 | calibration (1024 in / 1024 out) | TBD | TBD | TBD | 15.06² |
| JAX native bf16 | paper-shaped (280 out) | TBD | TBD | TBD | 8.24³ |

Spread (mean / p95) per row: TBD, from `perf/results/*.json`.

¹ 256-token canvas / 16 denoising steps; vLLM's internal per-step forward accounting to
be confirmed from its detailed bench output.
² Deterministic given fixed shapes: 1024 tokens / (4 canvases × (16 denoiser + 1
cache-append) = 68 full-transformer forwards). Excludes prefill and the 64 lightweight
self-conditioning `encode_logits` applies (embedding ops, not transformer passes — their
cost appears in wall-clock only). To be confirmed against run logs.
³ Scored on 280 delivered tokens over 2 full canvases (512 emitted, 34 forwards) — the
canvas-quantization penalty is deliberately included. On emitted tokens the figure would
be 15.06.

### Compile and warmup costs (recorded, not part of timed results)

| Stack × workload | JIT compile + first sample | Notes |
|---|---|---|
| JAX native × calibration | TBD | |
| JAX native × paper-shaped | TBD | |

## Comparison with published references

| Row | Hardware | Precision | Output | Latency ms | Gen tok/s | Tok/Step |
|---|---|---|---|---|---|---|
| **Ours: JAX native** | H100 SXM | bf16 | 1024 forced | TBD | TBD | 15.06 |
| **Ours: JAX native** | H100 SXM | bf16 | 280 forced | TBD | TBD | 8.24 |
| vLLM blog | H100 | FP8 | 1024 forced | — | 1,008 | ~16 |
| Fast-dDrive best (+SGLang) | H100 | not stated | ~280 structured | 665 | 608.5 | 4.93 |
| Fast-dDrive AR baseline | H100 | not stated | ~280 structured | 7,855 | 51.6 | 1 |

Confounds to keep in mind when reading this table (accepted by decision, see
[decision.md](./decision.md) §7 and §1):

- **Precision**: ours is bf16; the vLLM reference is FP8. At batch 1 the model is
  memory-bound, so FP8 weights alone plausibly account for ~1.5–2×. Pre-agreed
  disambiguation if inconclusive: a vLLM-bf16 run on the same box.
- **Model**: DiffusionGemma is a 26B-A4B text-only MoE; Fast-dDrive is a smaller
  driving VLM with image inputs and structured outputs. Cross-model absolute numbers are
  indicative only; the paper's own headline framing (speedup vs AR baseline of the same
  model) is out of scope here by decision §2.
- **Stack**: research-grade JAX vs optimized serving stacks (vLLM / SGLang), including
  the native sampler's double-apply self-conditioning per denoising step.

## Findings and implications for the future HD/vision-stack benchmark

> TBD after runs. Expected topics: measured JAX-vs-vLLM gap decomposition; compile-time
> magnitude; real cost of the `encode_logits` self-conditioning applies; canvas
> quantization penalty in practice; whether an ignore-EOS patch was needed and what it
> touched; checkpoint access path from a rental box.

## Reproduction

1. `perf/setup_vast.sh` on a fresh vast.ai H100 SXM 80GB instance (≥200GB disk).
2. `perf/bench_vllm.sh` — infra calibration; check the gate above.
3. `perf/bench_native.py` — sanity run, then calibration and paper-shaped workloads.
4. Outputs land in `perf/results/`, one JSON per (stack × workload) with full provenance.
