# Decisions

Benchmark-defining decisions for the DiffusionGemma batch-1 inference benchmark, in the
order they were made (2026-08-02). Each entry records the rejected alternatives and the
trade-off that decided it. Vocabulary: [CONTEXT.md](./CONTEXT.md). Overall plan:
[PLAN.md](./PLAN.md).

## 1. Benchmark subject: native `gemma.diffusion.Sampler` + `DIFFUSIONGEMMA_26B_A4B_IT`

The productionized, fully-jitted inference path is the subject; nothing else is timed.

- **Rejected: the hackable_diffusion (HD) adapter stack** (the sudoku-vision SFT path).
  It matches the comparison paper's vision-input shape and the ongoing project, but its own
  notebook states it is not optimized for inference speed — benchmarking it measures
  Kauldron/adapter overhead, not the model. Deferred, not abandoned: a future benchmark of
  this stack is planned, and transferable findings land in `LEARNINGS.md`.
- **Rejected: benchmarking both now.** Roughly double the work for a secondary number.

## 2. Comparability via infra calibration, not baseline replication

Before trusting any of our numbers, reproduce vLLM's published 1,008 generation tok/s
(H100, FP8, batch 1, their public repro gist) on the same rented box, within ~10%.

- **Rejected: loose factor-of-N comparison against the published number.** If our JAX
  figure lands 5× below 1,008, a loose comparison cannot separate "JAX tax" from broken
  infra. The repro turns infra validation into a controlled experiment: repro passes ⇒ the
  box is sound ⇒ the remaining gap is attributable to our stack.
- **Rejected (deferred): measuring an AR `Gemma4_26B_A4B` baseline for a diffusion-vs-AR
  speedup ratio.** That ratio is the only number genuinely comparable to Fast-dDrive's
  headline claims (their framing is "N× over AR"), but it doubles the run matrix. User
  chose diffusion-only numbers first.

## 3. Batch 1 only

No other batch sizes, anywhere. User requirement; both references (vLLM blog, Fast-dDrive)
are batch 1. Trade-off knowingly accepted: no serving-throughput story — public data shows
AR models overtake diffusion at batch ≳32 via KV-cache reuse, and this benchmark will not
observe that regime.

## 4. Two fixed-shape workloads

**Calibration**: random token IDs, 1024 in / 1024 out forced, 100 prompts — byte-for-byte
the vLLM gist shape, so the JAX-vs-vLLM gap is attributable. **Paper-shaped**: one fixed
real-text prompt repeated 100×, 280-token forced output — Fast-dDrive's per-sample output
shape.

- **Rejected: prompts drawn from a public instruct dataset (Dolly-100, fixed seed +
  pinned revision).** Would give a realistic prompt-length distribution and stay reusable
  when early stopping is later enabled, but variable sequence lengths cross JAX pad
  buckets (recompiles, bimodal latency) and — with early stopping off — prompt content
  cannot affect timing anyway. User ruled out variable shapes inside timed runs.
- **Rejected: relying on the timed runs to catch a misloaded checkpoint.** Split into a
  separate untimed sanity run (real prompt, output text saved and eyeballed) so timed
  inputs can stay content-free.

## 5. Canvas 256, 16 denoising steps, no early stopping, EOS ignored

- **Rejected: repo default sampler config (`max_denoising_steps=48` + entropy/
  token-stability early stopping).** 3× the forwards of the reference config and
  comparable to nothing published. Early stopping also makes step counts data-dependent
  (a batch runs until its slowest element), destroying determinism of the measurement.
  16 steps is the config behind the 1,008 tok/s reference; the comparison target defines
  the config.
- **Rejected: 32/64/96 step sweep (the repo's eval values).** Quality-eval territory, not
  a speed reference.
- EOS ignored to match the gist's `--ignore-eos`; if the native sampler lacks the switch,
  it gets a minimal local patch (documented in `LEARNINGS.md`).

## 6. Metrics: latency ms/sample, generation tok/s (median), end-to-end tok/s, Tok/Step

Definitions in [CONTEXT.md](./CONTEXT.md); chosen to be the union of what vLLM and
Fast-dDrive report, so every row is comparable to at least one published table.

- **Rejected: counting self-conditioning `encode_logits` applies as forward passes in
  Tok/Step.** They are embedding ops, not transformer passes; counting them would be
  misleading. They are footnoted instead (their cost still shows in wall-clock). Honest
  accounting: 256 tokens / (16 denoiser + 1 cache-append forwards) ≈ 15.1 nominal.
- **Rejected: scoring the paper-shaped row on 512 emitted tokens.** A 280-token request
  costs 2 full canvases (512 emitted). Scoring on 512 flatters the model and breaks
  Fast-dDrive's per-sample accounting; the canvas-quantization penalty ("pay for 512 to
  get 280") is a finding the benchmark exists to surface. Scored on 280 delivered.

## 7. JAX at bf16; FP8-vs-bf16 confound accepted and footnoted

At batch 1 the model is memory-bound (3.8B active params), so the reference's FP8 weights
alone plausibly buy ~1.5–2×; our bf16 number is expected to land below 1,008 even on
perfect infra.

- **Rejected (pre-agreed fallback): an extra vLLM run with the bf16 checkpoint.** Would
  cleanly decompose quantization tax (vLLM-FP8 vs vLLM-bf16) from stack tax (vLLM-bf16 vs
  JAX-bf16) for one ~52 GB download + minutes of bench time. To be executed only if the
  accepted-confound results prove inconclusive.
- **Rejected: FP8 inference in JAX.** Not supported by the repo's quantization utilities;
  days of work, out of scope.

## 8. Hardware: 1× H100 SXM 80GB, vast.ai on-demand, destroyed after

- **Rejected: H100 PCIe/NVL.** ~40% less memory bandwidth than SXM; at batch 1 the
  workload is memory-bound, so the variant choice alone would poison the calibration
  against the (SXM) reference.
- **Rejected: interruptible/spot instances.** Cheaper, but a mid-run preemption ruins
  timing and forces re-runs; total cost is only ~$10–20 on-demand.
- **Rejected: local CPU measurement.** No comparable reference exists; CPU is used only
  to smoke-test the harness before renting.
- Destroy, not stop, when done — stopped vast.ai instances keep billing storage.

## 9. Protocol: compile excluded but recorded; 3 warmups; 100 timed samples, single pass

Per-request host wall-clock around `jax.block_until_ready()`; report median (matches
vLLM's per-request median) plus mean and p95; one pad bucket per config; full provenance
(GPU, driver/CUDA, jax/vLLM versions, XLA flags, checkpoint id, all sampler knobs) in
every results JSON.

- **Rejected: including JIT compile in timed results.** It is a one-time cost (~1–2 min)
  that would swamp per-request statistics; recorded separately because compile time is
  itself a transferable finding.
- **Rejected: repeated sweeps per config.** 100 samples of a deterministic, fixed-shape
  workload already give a stable distribution.

## 10. Deliverables: `perf/` at repo top level, docs under `perf/docs/`

Scripts (`bench_native.py`, `bench_vllm.sh`, `setup_vast.sh`) and `results/` live in
`perf/`; all documents (this file, `PLAN.md`, `CONTEXT.md`, later `LEARNINGS.md`) live in
`perf/docs/`. Results JSONs are committed (following the repo's existing
`tools/rampup/measured/` precedent). Commits local-first; nothing pushed without explicit
request.

- **Rejected: `tools/perf/`.** User preference for a top-level `perf/` directory.
- **Rejected: ADRs.** None of the decisions above are hard to reverse — they are
  benchmark methodology, cheap to change — so a decision log beats the ceremony of
  one-file-per-decision.
