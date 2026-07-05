# Phase 1 notes — Model architecture (static exploration)

All paths relative to repo root. "gemma4" = `gemma/gm/nn/gemma4/`, "diffusion" = `gemma/diffusion/`.

## Class assembly

`DiffusionGemma_26B_A4B` (diffusion/_models.py:21-42) multiply-inherits:
- `_gemma4.Gemma4_26B_A4B` (gemma4/_gemma4.py:238-300) → `_Gemma4Base` (gemma4/_gemma4.py:36-60) → `Transformer` (gemma4/_transformer.py:110)
- `DiffusionMixin` (diffusion/_transformer.py:81) — adds `call_with_self_conditioning`.

`setup()` (diffusion/_models.py:33-42) calls `super().setup()` then adds `self.self_conditioner`.

## Config (gemma4/_gemma4.py:258-296; dataclass gemma4/_config.py:87-131)

| Field | Value | line |
|---|---|---|
| num_embed (V) | 262144 | _gemma4.py:259 |
| embed_dim (D) | 2816 | :260 |
| hidden_dim (dense FFN) | 2112 | :261 |
| num_heads | 16 | :262 |
| head_dim | 256 | :263 |
| num_kv_heads (local) | 8 | :264 |
| final_logit_softcap | 30.0 | :265 |
| num_global_kv_heads | 2 | :266 |
| qk_norm_with_scale | True | :269 |
| global_key_size | 512 | :271 (_DEFAULT_GLOBAL_KEY_SIZE :32) |
| k_eq_v_global | True | :272 |
| global_rope_proportion | 0.25 | :273 |
| local_rope_proportion | 1.0 | :274 |
| sliding_window_size | 1024 | :276 |
| local/global base freq | 10_000 / 1_000_000 | :277-278 |
| per_layer_input_dim | 0 (PLE disabled) | :279 |
| enable_moe | True | :281 |
| num_experts | 128 | :282 |
| expert_dim | 704 | :283 |
| top_k_experts | 8 | :284 |
| moe_dense_hidden_dim | 2112 | :285 |
| use_bidirectional_attention | 'vision' | :295 |
| kv_cache_sharing_config | None (default) | _config.py:115 |

- num_layers = len(attention_types) = 30 (_config.py:133-135; _NUM_LAYERS_GEMMA4_26B_A4B_MOE=30 at _gemma4.py:31).
- Attention pattern: (5× LOCAL_SLIDING, 1× GLOBAL) tiled ×5 (_gemma4.py:247-254, _config.py:41-52). GLOBAL layers = 5,11,17,23,29.
- text_only=True by default → vision/audio encoders dropped (_gemma4.py:44,50-60).
- "26B_A4B": ~26B total params MoE, ~4B active (top-8 of 128 experts + dense shared MLP). Interpretation from docstring _gemma4.py:238-245.

## Module tree

```
DiffusionGemma_26B_A4B (Transformer subclass; diffusion/_models.py:21)
├── embedder : Embedder (gemma4/_transformer.py:138; gemma4/_modules.py:60)
│     └── input_embedding [V=262144, D=2816] (_modules.py:74-78)  ← also tied output head (decode, _modules.py:127-137)
├── blocks[0..29] : Block name=layer_i (gemma4/_transformer.py:157-195; _modules.py:468)
│     ├── skip_scale [1] (_modules.py:499)
│     ├── pre_attention_norm : RMSNorm [D] (_modules.py:497)
│     ├── attn : Attention (_modules.py:516; class :201)
│     │     ├── q_einsum.w  local [16,2816,256] / global [16,2816,512] (_modules.py:225; global key_size _modules.py:511-512)
│     │     ├── kv_einsum.w local [2,8,2816,256]  (_modules.py:233-235)  | global k_einsum.w [2,2816,512] (k_eq_v, :229-231)
│     │     ├── attn_vec_einsum.w [16,256|512,2816] (_modules.py:222-224)
│     │     ├── query_norm.scale / key_norm.scale [256|512] (_modules.py:236-237); value_norm no scale (:238)
│     ├── post_attention_norm [D] (_modules.py:531-533)
│     ├── MoE FFN branch (_setup_moe, _modules.py:562-591):
│     │     pre_ffw2_norm [D] (:567); mlp2 : FeedForward dense hidden 2112 (:568, class :422)
│     │       ├── gating_einsum [2,2112,2816] (:441-444); linear [2112,2816] (:456-459)
│     │     post_ffw2_norm [D] (:572-574); pre_ffw_norm [D] (:577)
│     │     mlp : MoERagged (:578; gemma4/_moe.py:261)
│     │       ├── router_logits.w [2816,128] (_moe.py:274-276)
│     │       ├── gating_einsum.w [128,2,704,2816] (:282-284); linear.w [128,704,2816] (:286-288)
│     │       ├── per_expert_scale [128] (:289-293); router_scale [2816] (:294-298); router_norm no scale
│     │     post_ffw1_norm [D] (:584-586); post_ffw_norm [D] (:589-591)
├── final_norm : RMSNorm [D] (gemma4/_transformer.py:196)
└── self_conditioner : SelfConditioning (diffusion/_models.py:42; diffusion/_transformer.py:53)
      ├── pre_norm.scale [D] (diffusion/_transformer.py:60)
      ├── ffw : FeedForward feat=2816 hidden=2112 → gating_einsum [2,2112,2816], linear [2112,2816] (:61-64)
      └── post_norm (no scale, :65)
```

## Why "diffusion"

1. Attention mask supplied externally (not forced causal) via `call_with_self_conditioning` (diffusion/_transformer.py:86-166). Denoiser mask built by `mask_helpers.create_decoder_attention_mask` (mask_helpers.py:247-309): canvas tokens attend to non-pad prompt + all tokens of canvases with id ≤ selected — bidirectional within the denoised canvas, block-causal across canvases.
2. **Diffusion time t is NOT injected into the network.** `WrappedDiffusionGemmaNetwork.__call__` accepts `time` (hd_gemma_network.py:190) but never uses it (body :207-245). No AdaLN, no time embedding.
3. Self-conditioning: previous denoising step's logits fed back. `conditioning['sc_logits'] [B,L,V]` (hd_gemma_network.py:222-228) → `embedder.encode_logits` = softmax(logits)·table ·√D (_modules.py:139-155) → `sc_embeddings [B,L,D]` → `SelfConditioning`: `post_norm(canvas_emb + ffw(pre_norm(sc)))` replaces token embeddings (diffusion/_transformer.py:154-164). Zeroed when sc all-zero (first step).

## WrappedDiffusionGemmaNetwork (hd_gemma_network.py:115-245)

Implements `hackable_diffusion.lib.diffusion_network.BaseDiffusionNetwork` protocol (diffusion_network.py:77-87).
- `__call__(*, time, xt, conditioning, is_training)` → `{'logits': [B,L,V]}` (:186-245). xt `[B,L,1]`→ tokens `[B,L]` (:210-215). Prompt is NOT concatenated here — it lives in `conditioning['kv_cache']` (prefilled) + attention mask.
- `encoder_call(*, x, conditioning_embeddings)` (:153-184): standard causal `Transformer.__call__` over prompt(+x0) — prefill; returns full Output (logits + cache).
- `prefill_kv_cache_with_encoder` (:52-107); `init_cache` (:131-151) → per-layer `{k,v: [B,cache_len,n_kv,head_dim], end_index:[B], positions:[B,cache_len]}` (_modules.py:399-419, _config.py:157-193).

## Shape flow (one denoiser forward)

tokens [B,L] → embed ·√D [B,L,D] (_modules.py:112-125) → self-cond combine [B,L,D] (diffusion/_transformer.py:160-164) → 30× Block:
pre_norm → q einsum `BTD,NDH->BTNH` [B,L,16,256|512] (_modules.py:265) → kv `BSD,CKDH->CBSKH` 2×[B,L,8,256] local (:281) / k=v [B,L,2,512] global (:278) → RoPE (:267-273,289-295; gm/math/_positional_embeddings.py:23-75) → GQA attn logits [B,L,16,Cache] (:328-334) → attn_vec `BTNH,NHD->BTD` [B,L,D] (:375) → +residual → dense mlp2 + MoE mlp branches summed (:637-693) → ×skip_scale (:662)
→ final_norm [B,L,D] (_transformer.py:389) → embedder.decode x·tableᵀ [B,L,V] (_modules.py:137; diffusion/_transformer.py:180) → tanh softcap ×30 (diffusion/_transformer.py:182-184).

## Tiny model construction (from diffusion/_models_test.py:31-102)

```python
small_config = _config.TransformerConfig(
    num_embed=32, embed_dim=8, num_heads=2, num_kv_heads=1, head_dim=4,
    hidden_dim=16, attention_types=[_modules.AttentionType.GLOBAL],
    kv_cache_sharing_config=None, use_post_attn_norm=True, use_post_ffw_norm=True,
    final_logit_softcap=None, global_rope_proportion=1.0)
model = _models.DiffusionGemma_26B_A4B(config=small_config,
    self_conditioning_config=_transformer.SelfConditioningConfig(features=8, hidden_dim=16))
```
Gotchas: must init via `method=model.call_with_self_conditioning` with `sc_embeddings`; rope_proportion required per layer type; head_dim even; custom config disables MoE unless enable_moe=True explicitly; sft_sudoku_full.py import monkey-patches Block.__call__ with nn.remat (import side effect).

## Open questions
- `keep_last_prefill_kv=True` (diffusion/_models.py:31) — consumer not traced.
- Active-param count "A4B" is interpretation, not stated numerically in code.
