#!/usr/bin/env bash
# Bootstrap a fresh vast.ai H100 SXM 80GB box for the benchmark
# (perf/docs/plan.md; rental spec in decision.md section 8).
#
# Intended base image: vllm/vllm-openai:gemma (ships the vLLM diffusion build
# used by bench_vllm.sh). The native JAX stack goes into a separate venv so
# jax[cuda13]'s bundled CUDA wheels never mix with the image's torch/CUDA
# packages (AGENT.md: mixing CUDA 12/13 packages corrupts NCCL/PJRT).
#
# Everything downloads anonymously: gs://gemma-data is publicly readable
# (verified 2026-08-02; 40.4 GB orbax/ocdbt checkpoint, 32 objects), and the
# FP8 repro checkpoint is pulled from HF by vLLM at serve time (not gated).
#
# If the GitHub repo is private on this box, set REPO_URL to a tokened URL
# (https://<token>@github.com/...) or scp the tree to $WORKDIR/gemma first;
# an existing $WORKDIR/gemma is used as-is.
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/xinshengqin/gemma.git}"
REPO_BRANCH="${REPO_BRANCH:-worktree-perf-exec}"
BUCKET="https://storage.googleapis.com/gemma-data"
CKPT_PREFIX="checkpoints/diffusiongemma-26B-A4B-it"
CKPT_DEST="$WORKDIR/ckpt/diffusiongemma-26B-A4B-it"
TOKENIZER_DEST="$WORKDIR/ckpt/tokenizer_gemma4.model"
VENV="$WORKDIR/venv-jax"

echo "== hardware checks =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$gpu_name" in
  *H100*) ;;
  *) echo "WARNING: expected H100, got: $gpu_name" >&2 ;;
esac
cuda_version=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*')
if [ "${cuda_version%%.*}" -lt 13 ]; then
  echo "ERROR: driver reports CUDA $cuda_version; jax[cuda13] needs >= 13" >&2
  echo "(adapter README: CUDA 13 exactly; other versions cause NCCL errors)" >&2
  exit 1
fi
command -v vllm >/dev/null || echo "WARNING: vllm CLI not found; bench_vllm.sh needs the vllm/vllm-openai:gemma image or vllm >= 0.24" >&2

echo "== system packages =="
apt-get update -qq && apt-get install -y -qq --no-install-recommends git jq python3-venv >/dev/null

echo "== repo =="
if [ ! -d "$WORKDIR/gemma/.git" ]; then
  git clone --branch "$REPO_BRANCH" --single-branch "$REPO_URL" "$WORKDIR/gemma"
else
  echo "using existing $WORKDIR/gemma ($(git -C "$WORKDIR/gemma" rev-parse --short HEAD))"
fi

echo "== jax venv =="
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q -e "$WORKDIR/gemma"
"$VENV/bin/pip" install -q -U "jax[cuda13]"   # after gemma, per adapter README

echo "== checkpoint (40.4 GB, anonymous GCS, 8-way parallel) =="
# Sentinel is script-owned, written only after the count check passes: the
# bucket's own commit_success.txt is one of the downloaded objects, so using
# it as the sentinel would let a partially-failed run poison every re-run.
if [ -f "$CKPT_DEST/.download_complete" ]; then
  echo "already downloaded: $CKPT_DEST"
else
  mkdir -p "$CKPT_DEST"
  objects=$(curl -sS "https://storage.googleapis.com/storage/v1/b/gemma-data/o?prefix=$CKPT_PREFIX/&maxResults=1000&fields=items(name)" \
    | jq -r '.items[].name')
  echo "$objects" | xargs -P8 -I{} bash -c '
    obj="$1"; prefix="$2"; dest_root="$3"; bucket="$4"
    rel="${obj#"$prefix"/}"
    dest="$dest_root/$rel"
    mkdir -p "$(dirname "$dest")"
    # No plain -f: with -C - an already-complete file answers 416, which -f
    # turns into exit 22 and breaks re-runs. Check the HTTP code instead, and
    # delete on error so a saved error body cannot poison a later resume.
    code=$(curl -sS --retry 5 --retry-all-errors -C - -o "$dest" -w "%{http_code}" "$bucket/$obj") || code="curl:$?"
    case "$code" in
      2??|416) ;;
      *) echo "ERROR: download failed ($code): $obj" >&2; rm -f "$dest"; exit 1 ;;
    esac
  ' _ {} "$CKPT_PREFIX" "$CKPT_DEST" "$BUCKET"
  want=$(echo "$objects" | wc -l)
  have=$(find "$CKPT_DEST" -type f | wc -l)
  if [ "$want" != "$have" ]; then
    echo "ERROR: checkpoint download incomplete ($have/$want objects)" >&2
    exit 1
  fi
  touch "$CKPT_DEST/.download_complete"
  echo "downloaded $have objects to $CKPT_DEST"
fi

echo "== tokenizer =="
if [ ! -f "$TOKENIZER_DEST" ]; then
  code=$(curl -sS --retry 5 --retry-all-errors -o "$TOKENIZER_DEST" -w "%{http_code}" "$BUCKET/tokenizers/tokenizer_gemma4.model") || code="curl:$?"
  case "$code" in
    2??) ;;
    *) echo "ERROR: tokenizer download failed ($code)" >&2; rm -f "$TOKENIZER_DEST"; exit 1 ;;
  esac
fi

echo "== env =="
cat > "$WORKDIR/env.sh" <<'EOF'
# GPU env for the native JAX runs (hackable_diffusion_adapter README/AGENT.md:
# constant_folding pass disabled to prevent JIT-compile OOM/hangs; allocator
# left growable so the vLLM server and JAX never fight over preallocation).
export XLA_FLAGS="--xla_disable_hlo_passes=constant_folding"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
# The vllm-openai image exports LD_LIBRARY_PATH=/usr/local/cuda/lib64:..., which
# makes jax's cuda13 plugin dlopen the image's system cuBLAS, fail its version
# check, and silently fall back to CPU. jax's pip wheels bundle their own CUDA
# libs and the driver's libcuda.so.1 is ldconfig-visible, so drop it entirely.
unset LD_LIBRARY_PATH
# The version check also failed intermittently with a clean env (LD_DEBUG
# confirmed the correct bundled libs resolve, so the check itself is the
# problem). Skipping it is safe here; bench_native.py's gpu-backend assert
# remains the hard gate if plugin init genuinely fails.
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
EOF

echo "== verify jax sees the GPU =="
# shellcheck disable=SC1091
source "$WORKDIR/env.sh"
"$VENV/bin/python" - <<'EOF'
import jax
assert jax.default_backend() == 'gpu', jax.default_backend()
print('jax', jax.__version__, '->', [d.device_kind for d in jax.devices()])
import gemma.diffusion  # noqa: F401  (import check only)
print('gemma.diffusion importable')
EOF

echo "== provenance =="
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  nvidia-smi
  "$VENV/bin/pip" freeze | grep -Ei 'jax|gemma|kauldron|orbax|flax|hackable'
  command -v vllm >/dev/null && vllm --version || true
} > "$WORKDIR/provenance_setup.txt" 2>&1
echo "wrote $WORKDIR/provenance_setup.txt"

cat <<EOF

Setup complete. Next (plan.md execution order):
  1. bash $WORKDIR/gemma/perf/bench_vllm.sh          # infra gate vs 1008 tok/s
  2. source $WORKDIR/env.sh && \\
     $VENV/bin/python $WORKDIR/gemma/perf/bench_native.py sanity \\
       --checkpoint $CKPT_DEST --tokenizer $TOKENIZER_DEST
  3. ... bench_native.py calibration / paper (same flags)
Results land in $WORKDIR/gemma/perf/results/. Destroy (not stop) the instance
when they are pulled.
EOF
