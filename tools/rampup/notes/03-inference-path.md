# Phase 1 notes — Inference / AR diffusion sampling (static exploration)

## Which sampler is used
Eval uses **hackable_diffusion `hd.sampling.*`**, NOT gemma/diffusion/_sampler.py (dead code for this path). ar_eval.make_ar_evals builds hd.sampling.DiffusionSampler / DiffusionSamplerWithEarlyStopping + DiscreteDDIMStep (ar_eval.py:68-90), wrapped in sft_model.GemmaKDARSampler (subclass of hackable_diffusion AutoregressiveDiffusionSampler) at ar_eval.py:93-112.

Config: V=262144, prompt_len=256, num_canvases=1, canvas_size=256; uniform process (π=1/V, NOT masking), alpha(t)=1-t (RFSchedule). max_num_canvases=1 (eval_main.py:143, sft_sudoku_full.py:269) → cache_length = 256+1·256 = 512 (hd_gemma_ar_state_handler.py:195). Temperature 0.7, logits bf16, steps ∈ {32,64,96} (ar_eval.py:31,72-73), entropy threshold 0.05 (early-stopping variant).

## Call graph
eval_main.main (eval_main.py:164) → _inject_ar_evals (:126) → ar_eval.make_ar_evals (ar_eval.py:34) → evaluator.evaluate (eval_main.py:207) → CheckpointedEvaluator.evaluate (checkpointed_evaluator.py:127) → GemmaSamplingEvaluator._step (sft_model.py:491) → _ar_diffusion_step (sft_model.py:506-549) → GemmaKDARSampler.__call__ = AutoregressiveDiffusionSampler.__call__ (hackable_diffusion ar_diffusion_sampler.py:244) → per canvas: DiffusionSampler.__call__ (sampling.py:190) → per step: SFTInferenceFn.__call__ (sft_model.py:47) → WrappedDiffusionGemmaNetwork.__call__ (hd_gemma_network.py:186) → call_with_self_conditioning (diffusion/_transformer.py:86).

## Outer AR loop (ar_diffusion_sampler.py:244-332)
- while_loop: step < max_num_canvases AND not all done (:278-283); done via DoneEarlyStoppingFn = all(state['done']) (:188-196).
- Prefill: GemmaARStateHandler.init_ar_state (hd_gemma_ar_state_handler.py:170-270): prompt_tokens [B,256], prompt_lengths [B] (:192-194); input_mask (:199); prefill_kv_cache_with_encoder (:201) — one encoder_call writes prompt K/V; end_index=256 for all (not overridden, :88-90, hd_gemma_network.py:93-94). full_attention_mask [B,512] (make_full_attention_mask, mask_helpers.py:148). canvas_positions = prompt_lengths+arange(256) [B,256] (:224-226). canvas attn mask [B,256,512] via create_decoder_attention_mask(selected_canvas_idx=0) (:229-241).
- AR state (:256-269): prompt_tokens, prompt_lengths, prompt_mask, predicted_tokens [B,512] (prompt⧺zeros), step (init 256), done [B], kv_cache, positions [B,256], attention_mask [B,256,512], full_attention_mask [B,512], processed_denoising_steps, processed_num_canvases.
- Per canvas body (:285-320): x_T = sample_from_invariant → uniform random tokens [B,256,1] (:292; discrete.py:216-228); create_conditioning_from_state (:310; handler :403); canvas_sampler(...) inner loop (:306); update_ar_state (:315; handler :276-359):
  - xt squeeze [B,256] (:293-296); truncate_canvas_at_stop_tokens (:298,:431) — after first end token → PAD; end tokens = (EOS, END_OF_TURN, BEGIN_OF_TOOL_RESPONSE) (ar_eval.py:100-104); done |= has_stop (:308)
  - append_tokens_to_cache (:312,:470-521): causal forward pass writes canvas K/V at positions; mask = make_causal_attention_mask_right_pad(num_valid=end_index) & full_attention_mask (:501-508); advances end_index by 256
  - predicted_tokens[:, step:step+256] = canvas (:323-327); step += 256 (:329); positions += 256 (:332); rebuild attn mask selected_idx+1 (:339-352); counters (:354-357)
- finalize_ar_state → predicted_tokens[:, prompt_len:] [B, num_canvases·256] (:365-380); expand_dims → context.samples (sft_model.py:550).

## Inner diffusion loop (sampling.py:190-281; early-stop variant :321-457)
- Schedule: UniformTimeSchedule.all_step_infos (time_scheduling.py:96-121): time = linspace descending ~1→~0 (ε=1e-6), StepInfo{step, time [num_steps,B,1], rng}.
- scan over num_steps-1 + finalize (:214-281). Per step (:232-249):
  - PropagateSelfConditioningFn sets conditioning['sc_logits'] = prev step aux['logits'] (hd_gemma_ar_state_handler.py:47-64)
  - prediction = inference_fn(xt [B,256,1], conditioning, time [B,1]) → logits [B,256,V]
  - DiscreteDDIMStep.update (discrete_step_sampler.py:844-929) — D3PM-style 3-way routing (stay/noise/clean), NOT confidence unmasking:
    1. logits/temperature (0.7); x0 ~ categorical(logits) [B,256,1]; x_noise ~ uniform (:865, :370)
    2. alpha_s=1-t_next, alpha_t=1-t, ratio=alpha_t/alpha_s (:881-883)
    3. pi=1/V; p_stay=ratio(1-alpha_s)π, p_noise=(1-ratio)(1-alpha_s)π, p_clean=(1-ratio)·alpha_s·π (:888-893)
    4. x0==xt → merge clean into stay (:898-900)
    5. sample routing ~ categorical(log weights) → next xt from {xt, x_noise, x0} (:912, :334). Returns DiffusionStep{xt, step_info, aux={'logits'}} (:925-929). finalize = one more update (:931-942).
- Early stopping (DiffusionEntropyEarlyStopFn, hackable_diffusion diffusion_early_stopping.py:117-134): entropy = mean over tokens of -Σ softmax·log_softmax per [B]; stop when ≤ 0.05. Per-element freeze (sampling.py:412,:130); executed steps recorded (:446-455).

## Pivotal tensors
- KV cache: per layer {k,v: [B,512,n_kv,head_dim] (4-D!), end_index [B] int32 write cursor, positions [B,512]} (_modules.py:400-419, _config.py:157-193). Global layers: 2 heads, key 512; local: 8 heads, 256.
- xt: [B,256,1] int32 (trailing 1 enforced, discrete.py:396-402). aux logits [B,256,V] bf16.
- time: per step [B,1].
- predicted_tokens [B,512] int32; done [B] bool.

## Train vs inference call
| Aspect | Train | Inference |
|---|---|---|
| Cache | prefill prompt⧺full x0 (sft_encode), set_cache_end_index to prompt_len+idx·256 | prefill prompt only; canvases appended after each denoise |
| Canvas | random selected idx | sequential |
| Mask queries | all canvases (256·num_canvases) | one canvas (256) |
| Positions | from sft_encode slice | prompt_lengths+arange, +256/canvas |
| Self-cond | 2 passes, prob 0.5, stop-grad | every step from prev logits, zero first step |
| is_training | True | False |
| xt | corrupt(x0, t~U) | from uniform invariant, iterated |

## Open questions
- gemma/diffusion/_sampler.py + _early_stopping.py (threshold 0.005) are the unused parallel implementation.
- With num_canvases=1, multi-canvas machinery runs a single iteration.
