# Diffusion Gemma performance benchmarking

Vocabulary for measuring inference latency/throughput of Diffusion Gemma, calibrated against
the vLLM public numbers and comparable to Fast-dDrive (arXiv 2605.23163).

## Language

**Canvas**:
The fixed-size block of tokens (256) that one outer sampling iteration denoises in parallel.
_Avoid_: block, chunk

**Denoising step**:
One refinement iteration of a canvas in the inner sampling loop. Not the same as a model
forward pass — a step may cost more than one forward.
_Avoid_: sampling step, iteration

**Forward pass**:
One `model.apply` on the device. The unit for Tok/Step accounting.

**Generation tok/s**:
Decode-only tokens per second for a single request, reported as the median across requests
(vLLM's definition — the 1,008 H100 reference number is this metric).
_Avoid_: TPS, throughput (both ambiguous about prefill and aggregation)

**End-to-end tok/s**:
Total output tokens divided by full wall-clock time including prefill.

**Latency**:
Full wall-clock time per request in milliseconds, prefill included (Fast-dDrive's "latency (ms)").
_Avoid_: decode time, TTFT

**Tok/Step**:
Output tokens divided by the number of full-transformer forward passes during decode.
Lightweight non-transformer applies (e.g. self-conditioning logit encoding) are excluded
from the denominator and footnoted.
_Avoid_: tokens per step (ambiguous about what a step is)

**Calibration workload**:
The gist-exact vLLM benchmark shape: random-token prompts, 1024 in / 1024 out forced,
100 prompts, batch 1. Exists solely to attribute the JAX-vs-vLLM gap.

**Paper-shaped workload**:
One fixed real-text prompt repeated N times, 280-token forced output, batch 1, mirroring
Fast-dDrive's per-sample output shape. Fixed shapes only — no variable sequence or sampling
lengths inside any timed run.

**Sanity run**:
A single untimed generation on a real prompt whose output text is saved and eyeballed to
confirm the checkpoint loaded correctly. Not a benchmark; exists so timed runs can use
content-free inputs.

**Infra calibration**:
Reproducing vLLM's published 1,008 tok/s on our rented GPU before trusting any of our own
numbers. Passing means the hardware/driver stack is sound; a JAX gap is then attributable
to our code.
