# Math-vision SFT data plan

Sources for multimodal math SFT of the diffusion Gemma adapter. All datasets are
on HuggingFace and format-compatible (image + text).

Constraint driving the mix: no RL infra. The reasoning ceiling is therefore
fixed at SFT time, so the mix is CoT-dominant — answer-only data is the format
RL recovers reasoning from via rollouts, which is not available here.

## Train

| Dataset | HF path | Format | Size (usable) | Purpose in mix |
|---|---|---|---|---|
| **MAVIS-Instruct** | `lmms-lab/LLaVA-OneVision-Data` → configs `mavis_math_metagen` (87K) + `mavis_math_rule_geo` (38K) | Short CoT (~80–110 tok, prose) | ~125K | **Primary CoT base.** Field-standard, bounded length, survives denoising well; already includes geometry |
| **MultiMath-300K** | `pengshuai-rin/multimath-300k` (EN split) | Structured CoT (`Step N:` … `\boxed{}`, ~180–220 tok) | ~290K | **Structured-reasoning CoT.** Explicit numbered steps + boxed answer = clean supervision and easy answer extraction |
| **MathV360K** | `Zhiqiang007/MathV360K` | Answer-only | ~360K (sample down) | **The answer-only ~30% slice.** Keeps the model robust at emitting a clean final answer |
| Geo170K *(optional)* | `Luckyjhg/Geo170K` (qa split) | Brief derivations | ~118K | Skip unless extra geometry is wanted — MAVIS already covers it, and it carries the caption-hallucination caveat (VLAA-Thinker dropped it from SFT) |

**Mix:** ~70% CoT / 30% answer-only (MAmmoTH-VL's best-average ratio). Within the
CoT 70%, MAVIS as the base + MultiMath for longer structured steps. Subsample
MathV360K down to hit the 30% — don't let its raw 360K dominate.

## Test / held-out eval

| Dataset | HF path | Role | Notes |
|---|---|---|---|
| **MATH-Vision** | `MathLLMs/MathVision` — `testmini` (304) / `test` (3040) | **Primary target.** testmini for iteration, test for final | Eval-only, competition-sourced → clean held-out. **Has no train split; never train on it** |
| **MathVerse** | `AI4Math/MathVerse` (testmini) | Secondary reasoning eval | Cleaner vision-reasoning probe; low source overlap with the train mix |
| MathVista | `AI4Math/MathVista` (testmini, 1000) | Report with caveat | Shares seed datasets with MathV360K/MAVIS/Geo170K — **not a clean held-out test** here; treat as in-domain sanity check, not generalization |

## Implementation notes

- **Unify the final-answer marker** across sources before mixing (MultiMath uses
  `\boxed{}`, MAVIS uses "The answer is X", MathV360K varies). Pick one so answer
  extraction at eval is reliable.
- **De-contaminate:** MathVista and the MathV360K/MAVIS/Geo170K train data draw
  from overlapping seeds, so MathVista gains are not generalization. MATH-Vision
  and MathVerse are the real generalization signals.
- **Response length budget:** MultiMath CoT ≈ 180–220 tokens, MAVIS ≈ 80–110.
  Size the canvases accordingly, and prefer block / semi-autoregressive decoding
  so reasoning tokens condition the answer tokens.
- **Avoid long-CoT corpora** (e.g. MMathCoT-1M's distilled traces) — diffusion
  LMs degrade on long coherent generation more than AR models.
