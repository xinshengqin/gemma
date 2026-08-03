# Learnings

Major mistakes made and/or failures recovered over the course of this benchmark project.
Each entry: what went wrong, why, how it was caught, and how it was recovered. Final
results belong in [report.md](./report.md); decisions and their trade-offs belong in
[decision.md](./decision.md) — this file is only for things that went wrong.

## Entries

## 2026-08-02 — JAX silently benchmarked on CPU after cuBLAS version check failed

- **What went wrong**: the first M4 launch on the rental box ran the sanity
  mode on CPU. The `vllm/vllm-openai:gemma` image exports
  `LD_LIBRARY_PATH=/usr/local/cuda/lib64:...`; jax-cuda13's plugin dlopen'd the
  image's system cuBLAS, failed its version check ("Outdated cuBLAS
  installation found"), and jax fell back to CPU with only a warning —
  `bench_native.py` then happily started a 26B generation on 24 vCPUs.
- **Why it happened**: two ML stacks on one image; jax's pip wheels bundle
  their own CUDA 13 libs and must not see the image's. `setup_vast.sh`'s own
  GPU verify had passed in its earlier session, hiding the conflict (why that
  session escaped it was not pinned down; the fix removes the env dependence
  either way).
- **How it was caught**: the log watcher tripped on the Traceback ~2 min in;
  `nvidia-smi` showed 0 MiB used while python burned CPU.
- **Recovery**: killed the chain; verified the driver's `libcuda.so.1` is
  ldconfig-visible so the variable is unnecessary for jax; added
  `unset LD_LIBRARY_PATH` to the generated `env.sh` and a hard
  `jax.default_backend() == 'gpu'` assert to `bench_native.py`'s real-model
  path; relaunched. Cost: ~10 min of rental.
- **What to do differently** (esp. for the future HD/vision-stack benchmark):
  assert the accelerator backend at harness startup instead of trusting plugin
  discovery; on serving images, treat inherited env — `LD_LIBRARY_PATH` above
  all — as hostile to a second ML stack.

<!-- Entry template:
## YYYY-MM-DD — <one-line title>
- **What went wrong**:
- **Why it happened**:
- **How it was caught**:
- **Recovery**:
- **What to do differently** (esp. for the future HD/vision-stack benchmark):
-->
