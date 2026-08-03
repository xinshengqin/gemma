#!/usr/bin/env bash
# vLLM FP8 infra-calibration repro (perf/docs/plan.md, decision.md section 2).
#
# Protocol source: LucasWilkinson's repro gist behind the vLLM blog's
# 1,008 generation tok/s H100 batch-1 number. Serve + bench flags are verbatim
# from the gist; the only additions are --gpu-memory-utilization 0.85 and
# --generation-config vllm from the official deployment recipe (the
# checkpoint's generation_config.json would otherwise cap max_tokens at 256).
#
# Requires a vLLM build with diffusion support (docker image
# vllm/vllm-openai:gemma, or vllm >= 0.24). Gate: median generation tok/s
# within ~10% of 1008, else debug the box before any JAX runs.
set -euo pipefail

MODEL="${MODEL:-RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic}"
PORT="${PORT:-8000}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
RESULTS_DIR="${RESULTS_DIR:-$(dirname "$0")/results}"
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=4
GPU_MEM_UTIL=0.85
DIFFUSION_CONFIG='{"canvas_length":256,"max_denoising_steps":16}'
HF_OVERRIDES='{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1,"diffusion_confidence_threshold":0.0}'
GIT_COMMIT="$(git -C "$(dirname "$0")" rev-parse HEAD 2>/dev/null || echo unknown)"

mkdir -p "$RESULTS_DIR"
RAW_JSON="$RESULTS_DIR/vllm_fp8_calibration_raw.json"
OUT_JSON="$RESULTS_DIR/vllm_fp8_calibration.json"
SERVE_LOG="$RESULTS_DIR/vllm_serve.log"

echo "[vllm] starting server for $MODEL (log: $SERVE_LOG)"
vllm serve "$MODEL" \
  --port "$PORT" \
  --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" --trust-remote-code \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --generation-config vllm \
  --diffusion-config "$DIFFUSION_CONFIG" \
  --hf-overrides "$HF_OVERRIDES" \
  >"$SERVE_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Readiness poll, verbatim gist protocol (240 x 5s).
for i in $(seq 1 240); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health")" = 200 ] && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[vllm] server died during startup; tail of $SERVE_LOG:" >&2
    tail -30 "$SERVE_LOG" >&2
    exit 1
  fi
  sleep 5
done
echo "[vllm] server ready"

vllm bench serve --backend vllm --base-url "http://localhost:$PORT" \
  --model "$MODEL" --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --ignore-eos --num-prompts "$NUM_PROMPTS" --max-concurrency 1 \
  --save-result --save-detailed --result-filename "$RAW_JSON"

kill "$SERVER_PID" 2>/dev/null || true
trap - EXIT

# Generation tok/s per the gist: decode-only, per-request
# (output_len - first_chunk) / sum(itls) with first_chunk = one canvas (256)
# for diffusion, then the median across requests. This is NOT the printed
# "Output token throughput" line (that one includes prefill).
python3 - "$RAW_JSON" "$OUT_JSON" "$MODEL" "$NUM_PROMPTS" \
  "$DIFFUSION_CONFIG" "$HF_OVERRIDES" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" \
  "$GPU_MEM_UTIL" "$GIT_COMMIT" <<'PY'
import json, statistics, subprocess, sys
from datetime import datetime, timezone

(raw_path, out_path, model, num_prompts, diffusion_config, hf_overrides,
 max_model_len, max_num_seqs, gpu_mem_util, git_commit) = sys.argv[1:11]
raw = json.load(open(raw_path))
diffusion_config = json.loads(diffusion_config)
hf_overrides = json.loads(hf_overrides)
CANVAS = diffusion_config['canvas_length']

per_request = []
for output_len, itls in zip(raw['output_lens'], raw['itls']):
    if sum(itls) > 0:
        per_request.append((output_len - CANVAS) / sum(itls))
assert len(per_request) == int(num_prompts), (len(per_request), num_prompts)
per_request.sort()
median = statistics.median(per_request)
p95 = per_request[int(0.95 * (len(per_request) - 1))]

def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except Exception:
        return None

reference = 1008
result = {
    'stack': 'vllm_fp8',
    'workload': 'calibration',
    'config': {
        'model': model,
        **diffusion_config,
        **hf_overrides,
        'max_model_len': int(max_model_len),
        'max_num_seqs': int(max_num_seqs),
        'gpu_memory_utilization': float(gpu_mem_util),
        'generation_config': 'vllm',
        'random_input_len': 1024,
        'random_output_len': 1024,
        'ignore_eos': True,
        'num_prompts': int(num_prompts),
        'max_concurrency': 1,
    },
    'provenance': {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'vllm_version': sh(['vllm', '--version']),
        'gpu': sh(['nvidia-smi', '--query-gpu=name,driver_version',
                   '--format=csv,noheader']),
        'cuda_version': next(
            (line.split('CUDA Version:')[1].strip(' |').strip()
             for line in (sh(['nvidia-smi']) or '').splitlines()
             if 'CUDA Version' in line),
            None,
        ),
        'git_commit': git_commit,
        'raw_result': raw_path.split('/')[-1],
    },
    'metrics': {
        'generation_tok_s': {
            'median': median,
            'mean': statistics.mean(per_request),
            'p95': p95,
        },
        'output_token_throughput_e2e': raw.get('output_throughput'),
        'median_ttft_ms': raw.get('median_ttft_ms'),
        'median_tpot_ms': raw.get('median_tpot_ms'),
    },
    'gate': {
        'reference_tok_s': reference,
        'measured_median_tok_s': median,
        'ratio': median / reference,
        'pass_within_10pct': abs(median / reference - 1) <= 0.10,
    },
}
json.dump(result, open(out_path, 'w'), indent=2)
print(f"[vllm] median generation tok/s: {median:.1f} "
      f"(reference {reference}, ratio {median / reference:.3f}) -> "
      f"{'PASS' if result['gate']['pass_within_10pct'] else 'FAIL'}")
print(f"[vllm] wrote {out_path}")
PY
