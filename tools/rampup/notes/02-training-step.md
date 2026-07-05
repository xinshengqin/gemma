# Phase 1 notes — Training step (static exploration)

Constants (configs/sft_sudoku_full.py:109-117,227): B=8, prompt_len=256, num_canvases=1, canvas_size=256, total_canvas_len=256, full_seq_len=512, V=262144. PAD=0 (sft_model.py:37). Corruption: uniform CategoricalProcess + RFSchedule alpha(t)=1-t (hackable_diffusion schedules.py:176-177).

## Data pipeline (sudoku_data.make_sudoku_ds, sudoku_data.py:154-234)

Bagz file of tf.train.Example {puzzle, solution} (sudoku_data.py:99-107), PicklableBagzReader (:33-92), batch=8.

Transforms:
1. ParseSudokuExample (:95) → {prompt, response, puzzle_raw} strings
2. FormatText response (:167), prompt with chat template (:172; template sft_sudoku_full.py:130-136)
3. Tokenize prompt (add_bos=True) + response (:177-178)
4. Pad prompt to 256, truncate=False (:189)
5. CanvasChunker (data.py:37, called :205) → canvas, canvas_id, canvas_mask [256]
6. SequenceTargetShift (data.py:105, called :216) → encoder_target, encoder_target_mask [512]
7. Rearrange canvas "c -> c 1" (:220)
8. Elements keep (:234); eval also keeps solution_tokens/puzzle_tokens [256] (:195-201)

Final batch:
| key | shape | dtype |
|---|---|---|
| prompt | [B,256] | int32 |
| canvas | [B,256,1] | int32 |
| canvas_id | [B,256] | int |
| canvas_mask | [B,256] | bool |
| encoder_target | [B,512] | int32 |
| encoder_target_mask | [B,512] | bool |

CanvasChunker (data.py:64-102): response truncated to 256, PAD-filled flat array (:74-75); last real canvas EOS-filled to boundary (:80-83); canvas_id = repeat(arange(num_canvases), canvas_size) (:91); canvas_mask true for valid canvases (:96-97).

SequenceTargetShift (data.py:121-152): full_seq = concat(prompt, canvas) [512]; encoder_target = roll(full_seq,-1), last=PAD (:138-139); encoder_target_mask = full_valid & roll(full_valid,-1) (:146-148). Classic AR next-token targets over prompt+response.

## SFTDiffusion.__call__ (sft_model.py:276-431)

A. time = time_sampler(rng('sampling'), x0) → [B,1,1] in [1e-4, 1-1e-4] (:307; UniformTimeSampler axes=(0,), time_sampling.py:121-125)
B. corrupt (:310-312; discrete.py:230-311): alpha=1-t; keep mask ~Bernoulli(alpha) [B,256,1] (:277-284); noise ~Uniform(V) (:287); xt = where(keep, x0, noise) [B,256,1] (:290). target_info = {x0 [B,256,1], logits=one_hot [B,256,V], is_corrupted, is_unused} (:302-309).
C. selected_canvas_idx ~ U[0, num_valid_canvases) [B] (:321-334); x0_tokens = x0[...,0] [B,256] (:337).
D. sft_encode (:159-217, called :343-353):
   - full_seq = concat(prompt, x0_tokens) [B,512] (:197); full_seq_mask [B,512] (:199-200)
   - prefill_kv_cache_with_encoder (hd_gemma_network.py:52-107): positions=cumsum(mask)-1 [B,512] (mask_helpers.build_positions_from_mask :69-88); causal prefill mask [B,512,512] (make_causal_prefill_mask, mask_helpers.py:91-140); gemma_model(...) → encoder_logits [B,512,V] + prefilled kv_cache
   - end_index = prompt_len + selected_idx*canvas_size [B]; set_cache_end_index (mask_helpers.py:317-344; sft_model.py:211-215)
   - stop_gradient_from_denoiser_to_encoder=False → grads flow into encoder (:355-356)
E. sft_decode pass 1 (:90-155, called :377):
   - attention_mask = create_decoder_attention_mask(...) [B,256,512] (mask_helpers.py:247-309): canvas query attends non-PAD prompt OR canvas kv with id ≤ selected & valid. Bidirectional within selected canvas; future canvases hidden.
   - canvas_positions = positions[:, prompt_len:] [B,256] (:140)
   - gemma_network(xt, time, conditioning) → logits [B,256,V]
F. target_mask = canvas_mask & (canvas_id == selected[:,None]) [B,256] (:380); into target_info as [B,256,1] (:383-386); convert_predictions → {x0=argmax, logits} (discrete.py:313-331; :389)
G. self-conditioning (:397-418): sc_logits = stop_grad(pass1 logits); per-example Bernoulli(self_cond_prob=0.5, default sft_model.py:270) → zero or keep (:402-411); sft_decode pass 2 with sc_logits (:413) → FINAL logits used by loss; noise_info = schedule.evaluate(time) {time,alpha,sigma,logsnr each [B,1,1]} (:421).

Return dict → context.preds (:423-431):
- output {x0 [B,256,1], logits [B,256,V]} (2nd pass)
- target {x0, logits, is_corrupted, is_unused, target_mask [B,256,1]}
- xt [B,256,1]; noise_info; encoder_logits [B,512,V]; encoder_target [B,512]; encoder_target_mask [B,512]

## Losses (sft_sudoku_full.py:172-186, both weight 1.0)

diffusion_loss: KauldronLossWrapper (kdiff/core.py:174-193, keys preds.output/preds.target/preds.noise_info.time) → NoWeightDiscreteLoss.compute_discrete_diffusion_loss (discrete_loss.py:41-122): labels = target.x0.squeeze [B,256] (:60); mask = target_mask.squeeze (:62-68); plain CE -softmax_cross_entropy_with_integer_labels(logits, labels, where=mask) [B,256] (:74-78); normalize by per-example sum(mask) → [B] (:87-122). NO schedule/SNR weighting. x0-prediction CE over selected-canvas valid tokens (clean and corrupted alike).

encoder_loss: EncoderARLoss (sft_model.py:600-643): CE(encoder_logits, encoder_target) [B,512] × mask, per-example mean → [B]. The "encoder" = causal AR prefill pass predicting next token over prompt+clean-response; trains backbone AR ability while its KV conditions the denoiser.

Aggregation: AllReduceMean (kauldron losses/base.py:47-70) batch-mean each, total = 1.0·diffusion + 1.0·encoder (compute_losses, losses/base.py:261-281).

## One kauldron train step (kauldron train_step.py:211-329)

1. rngs = rng_streams.train_rngs(step) (:305); streams default+sampling (sft_sudoku_full.py:214-217); model uses 'sampling' (sft_model.py:307,311,328,404)
2. Context.from_state_and_batch (:300)
3. forward inside jax.grad(forward_with_loss, argnums=0, has_aux=True) (:292-307); model.apply with kontext-resolved args (:337-379); returned dict → context.preds (:359-378)
4. compute_losses → scalar total (:382-417; losses/base.py:261-281)
5. grads = context_grads.params (:301-308)
6. optimizer.update → apply_updates (:309-312). named_chain order (sft_sudoku_full.py:200-205): clip_by_global_norm(1.0) → scale_by_factored_rms (adafactor) → add_decayed_weights(1e-4) → scale_by_learning_rate(warmup_cosine: 0→1.5e-4→1.5e-5, warmup 1000, decay 2000)
7. state update step+1 (:314-319); FSDP sharding params+opt (sft_sudoku_full.py:138-140). Block remat monkey-patch (sft_sudoku_full.py:31-81).

Cost per step ≈ 1 encoder forward + 2 denoiser forwards (self-conditioning double pass).

## Open questions
- self_cond_prob=0.5 implicit (not in config).
- is_corrupted computed but unused by loss (mask_key="target_mask").
- Loss covers clean+corrupted tokens (uniform process x0-prediction) — design choice.
- Pad truncate=False: over-length prompts unhandled (assumed fits).
