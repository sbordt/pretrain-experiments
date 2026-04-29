# Unlearning Method Hyperparameters

Reference hyperparameters from the original papers, for reproducing each unlearning
method on the OLMo-2 179M / 546M / 1B continued-pretraining setup.

## Gradient Ascent — Jang et al., ACL 2023

**Paper:** *Knowledge Unlearning for Mitigating Privacy Risks in Language Models*
- ACL Anthology: https://aclanthology.org/2023.acl-long.805/
- arXiv: https://arxiv.org/abs/2210.01504
- Reference implementation: https://github.com/joeljang/knowledge-unlearning

### Optimization

| Setting              | Value                                                     | Source                                  |
| -------------------- | --------------------------------------------------------- | --------------------------------------- |
| Optimizer            | **Adam** (`FusedAdam` under DeepSpeed, else `torch.optim.Adam`) | `models/Neo_Model.py:664–676` in repo |
| Learning rate        | **5e-5**, constant (no warmup, no LR schedule)            | §4.1 "Configurations"                   |
| Weight decay         | 0                                                         | §4.1                                    |
| Dropout              | 0                                                         | §4.1                                    |
| Loss                 | plain CE × −1 (sign flip)                                 | `training_step`: `return loss * -1`     |
| Mixed precision      | fp16                                                      | `configs/example.json`                  |
| Batch size           | **= s** (# samples forgotten at once); smaller-than-*s* batches degrade utility | §4.1 |
| Forget-sample length | 200 tokens                                                | §4.1                                    |
| Eval cadence         | every epoch                                               | `configs/example.json`                  |

### LR sweep grid (Appendix C, Fig. 9)

Tested on GPT-Neo 1.3B with s=32 and a 10-epoch budget:

`{1e-4, 8e-5, 5e-5, 3e-5, 1e-5}`

- Higher LR → forgets faster but utility collapses.
- 1e-5 → fails to hit the forgetting threshold within 10 epochs.
- **5e-5 is the chosen sweet spot.**

### Number of epochs (Table 3, s=32, mean over 5 seeds)

Two stopping protocols:
- **+UL**: stop when forget set is "no worse than the OPT baseline" (modest target).
- **+UL⁺**: stop when forget set is below the **Forgetting Threshold** (full forgetting).

| Model         | Epochs to "+UL" | Epochs to "+UL⁺" |
| ------------- | --------------- | ---------------- |
| GPT-Neo 125M  | 11.0            | 17.2             |
| GPT-Neo 1.3B  | 8.0             | 13.8             |
| GPT-Neo 2.7B  | 5.4             | 10.8             |

Scaling rule the paper highlights: **larger models forget faster** at the same LR.
Linear-extrapolating from these:
- **OLMo-2 1B**: ≈ 8–14 epochs to threshold.
- **OLMo-2 546M**: ≈ 14–17 epochs to threshold.
- **OLMo-2 179M**: ≈ 17+ epochs to threshold.

### Stopping criterion — Forgetting Threshold (Table 2)

Stop when forget-batch averages of **both** EL₁₀ and MA fall below thresholds derived
from 10,000 length-200 instances drawn from Pile-validation (with the Pile-train domain
distribution). EL = Extraction Likelihood at n=10; MA = Memorization Accuracy. Both
computed via naïve greedy decoding.

| Model         | EL₁₀ threshold | MA threshold |
| ------------- | -------------- | ------------ |
| GPT-Neo 125M  | 4.99 %         | 29.94 %      |
| GPT-Neo 1.3B  | 5.68 %         | 33.27 %      |
| GPT-Neo 2.7B  | 5.53 %         | 34.02 %      |

For OLMo-2, recompute thresholds on Dolma-validation rather than Pile-validation.

### Forget-set chunking (Section 5.1, Fig. 3)

- s ≤ 32 → negligible utility degradation across all sizes.
- s = 128 → substantial degradation regardless of model size.
- For larger forget sets: split into chunks of ≤ 32 and unlearn **sequentially**
  until each chunk hits the threshold. Chunks once forgotten stay forgotten;
  later chunks converge in 1–2 epochs.

### Drop-in defaults for `gradient_ascent.py` (mirroring `gradient_noise.py`)

- Optimizer: Adam, `weight_decay=0`, no warmup, constant LR.
- LR: **5e-5** primary; sweep `{1e-5, 3e-5, 5e-5, 8e-5}` as the single knob.
- Global batch size = forget-chunk size (≤ 32 samples × 200 tokens).
- Max 20 epochs; early-stop on EL₁₀ / MA crossing thresholds calibrated on Dolma-val.
- Eval every epoch.
- Full Exp-Dataset coverage: chunk into ≤ 32-sample batches and run sequentially.

### Caveat for our setting

Jang's LR = 5e-5 is calibrated for ~5–17 epochs over 32 × 200 tokens of post-hoc
unlearning on a fully-pretrained checkpoint — only a few hundred optimizer steps
total. Our gradient-noise sweep runs at **5e-6 / 7.5e-6 / 3.5e-6** over a much longer
continued-pretraining horizon (10 k steps). So 5e-5 is the right starting point for a
*Jang-style short, sharp* unlearn pass; if the longer horizon used for the noise
sweep is kept, bias the LR sweep one decade lower (e.g. `{1e-6, 3e-6, 5e-6, 8e-6}`).

---

## Goldfish Loss — Hans et al., NeurIPS 2024

**Paper:** *Be like a Goldfish, Don't Memorize! Mitigating Memorization in Generative LLMs*
- OpenReview: https://openreview.net/forum?id=DylSyAfmWs
- arXiv: https://arxiv.org/abs/2406.10209
- Reference implementation: https://github.com/ahans30/goldfish-loss (fork of LitGPT)

> Note: Goldfish is a **prevention-time** method (applied during pretraining), not a
> post-hoc unlearning method. It belongs in this file because it is the only
> single-knob memorization-mitigation method on the shortlist that has been
> validated in a from-scratch / continued-pretraining regime at 1B–7B scale.

### Method recap

Standard CE forward pass on all tokens, but the loss is computed only on a subset.
For each token position, drop from the loss with probability **1/k** (k = drop
frequency, the only knob). The model still conditions on the full context — only
the supervision is masked. Three mask choices are tested; the paper recommends and
uses the **hashed mask** as default.

### Optimization (Appendix A, "Hyperparameters")

All pretraining HPs are inherited from TinyLLaMA. Same settings used for both
1B and 7B variants (with the exceptions noted below):

| Setting          | Value                                                   | Source              |
| ---------------- | ------------------------------------------------------- | ------------------- |
| Optimizer        | **Adam**                                                | App. A              |
| Weight decay     | **0.1**                                                 | App. A              |
| Batch size       | **2 M tokens** (= 1028 samples × block size 2048)       | App. A              |
| Block / seq len  | 2048                                                    | App. A              |
| Peak LR          | **4e-4**                                                | App. A              |
| Min LR (cosine)  | 4e-5                                                    | App. A (1B run)     |
| LR schedule      | **Cosine** (1B) / **constant, no warmup** (7B-extreme)  | App. A              |
| Warmup steps     | **1000** (1B) / **0** (7B-extreme)                      | App. A              |
| Total steps      | 9536 (1B, 20B tokens) / 100 (7B-extreme, 100 docs × 100 epochs) | App. A      |
| Mixed precision  | LitGPT default                                          | repo                |

### The single knob: drop frequency *k*

Tested values: **k ∈ {3, 4}** as primary; comparisons also against k → ∞
(standard loss) and the random-mask / static-mask baselines.

- **k = 4** is the headline default ("4-GL"); used everywhere unless stated.
- **k = 3** = stronger forgetting, slightly more utility cost.
- Smaller *k* → less memorization but more supervised tokens lost → larger
  pretraining slowdown.
- Goldfish is equivalent to standard CLM in the limit of large *k*.
- For **low-entropy or sensitive content** (e.g. code, PII-rich docs) the paper
  recommends **higher mask rates than k=3/4** in the limitations section, but
  doesn't quantify.

### Mask strategy (Section 3.1–3.2)

Three options; main results use **hashed mask**:

| Strategy     | Rule                                        | Robust to duplicates? | Recommended? |
| ------------ | ------------------------------------------- | --------------------- | ------------ |
| Static       | Drop every *k*-th position                  | No (mask aligned to seq, not content) | No  |
| Random       | i.i.d. Bernoulli(1/*k*)                     | No (different mask each epoch → eventually leaks) | No |
| **Hashed**   | Mask token *xᵢ* iff hash(*xᵢ₋ₕ*…*xᵢ₋₁*) < 1/*k* | **Yes** (same n-gram → same mask) | **Yes** |

### Hash context width *h* (Section 3.2)

- Default: **h = 13** — same length used to flag train/test contamination in
  Brown 2020 (GPT-3) and GLaM, so 13-grams are content the authors *want* never
  memorized.
- Trade-off:
  - *h* too small (e.g. 7) → important (h+1)-grams may never be learned
    (paper's example: `"the Los Angeles Department of Water and Power"`).
  - *h* too large → first *h*−1 tokens of each document are mask-undetermined.
- **Normalize text before hashing** (soft dashes, non-breaking spaces, etc.) so
  near-duplicates with cosmetic variation hash identically. The paper points to
  the normalization scheme from Kirchenbauer et al. watermarking as a working
  reference.

### Compute / token budgeting

Goldfish supervises only **(1 − 1/k)** of input tokens, so it sees fewer
supervised tokens per step than standard training:

> supervised_tokens = (1 − 1/k) × input_tokens

To match a standard model's final validation loss with k-GL, multiply the
**input** token budget by **k/(k−1)** (= 1.333× for k = 4) so supervised-token
counts line up. Same end-state val loss, ~33 % more forward-pass compute at k=4.

### Stopping criterion / evaluation

No "forgetting threshold" — Goldfish is applied throughout training. The paper
evaluates verbatim memorization with:
- **Prefix length 32** tokens, **suffix length 32** tokens, greedy decoding (T=0).
- **RougeL** and **Exact-Match rate** on the suffix.
- Compared against a "control" model that never saw the canaries.

For our setting the analog is the existing
`evaluation/train-once-answer-all/verbatim_memorization.py` and
`prompt_extraction.py` evals already in the suite.

### Drop-in defaults for an `experiments`-level Goldfish hook

Goldfish is a **data-layer** intervention (token mask before CE reduction),
unlike `gradient_noise.py` which intervenes at the optimizer step. The natural
implementation point is in the framework's loss path, gated on a config flag —
analogous to Jang's `negative_loss` flag.

- Mask strategy: **hashed**
- *k*: **4** primary; sweep `{3, 4}` if budget permits.
- *h*: **13** (start here; only revisit if (h+1)-gram corruption is observed).
- Hash text-normalization: lowercase + Unicode NFKC + strip soft-dashes /
  non-breaking spaces (matches Kirchenbauer watermarking norm).
- Optimizer: existing OLMo-2 Adam config; no LR change needed (Goldfish does not
  re-tune LR vs. baseline).
- Token budget: scale **input** tokens by **k/(k−1)** = 1.333× for k=4 if the
  goal is val-loss parity with the no-Goldfish baseline.
- Apply selectively (paper §6): per-document opt-in is fine, e.g. only on the
  Exp-Dataset insertion documents while leaving Dolma untouched.

### Caveats for our setting

- Goldfish was validated on **from-scratch / continued pretraining of LLaMA-2 7B
  on canaries** and **TinyLLaMA-1B from scratch on RedPajama+canaries**, both
  with relatively heavy duplication (50× and 100× canary repetition). That's a
  closer match to your insertion regime than any post-hoc unlearning paper.
- The 1B run uses cosine to 4e-5 across 9536 steps over 20 B tokens; our 1B
  unlearning runs are *continuations* of OLMo-2 from step 100 000 over only 10 k
  more steps, so the LR/schedule HPs above are **not directly portable**. Keep
  the existing OLMo-2 LR schedule and only change the loss-mask layer.
- Goldfish does **not** prevent membership-inference attacks — only verbatim
  extraction. If the eval suite reports MIA-style metrics, expect them to look
  similar to the no-Goldfish baseline.
- Beam-search extraction with k=3 is robust; k≥4 starts to leak under beam-30
  attacks (paper §5.2). If you sweep k upward, the beam-search arm of
  `prompt_extraction.py` is the relevant safety net.

---

# Hyperparameter selection plan: 179M → 546M → 1B

This section is the operational plan for picking unlearning hyperparameters
across the four post-hoc methods now in the library — **GA**, **SimNPO**,
**RMU**, **LUNAR** — analogous to how the gradient-noise sweep used the 179M
result as the anchor and transferred via σ ∝ 1/√N to 546M/1B.

The shape of the plan is the same for each method:

1. Run a small **anchor sweep at 179M** over the 1–3 most sensitive knobs.
2. Pick the winner against a fixed **eval-bundle** (forget metric + retain
   constraint) using the existing eval scripts.
3. **Transfer** that winner to 546M and 1B via a method-specific rule, then
   run a **tightening sweep** of ±0.5 decade (or one neighboring depth-tier)
   to confirm.

What differs across methods is *which* knobs are scale-sensitive: GA and
SimNPO have a single scalar each (LR / β); RMU and LUNAR add a depth choice
(target/redirection layer) and, for RMU, an activation-magnitude choice
(steering coefficient `c`).

## Architecture quick reference

| size | layers | hidden | FFN  | global_batch (stage1) |
| ---- | ------ | ------ | ---- | --------------------- |
| 179M | 12     | 576    | 2304 | 512                   |
| 546M | 16     | 1120   | 4480 | 512                   |
| 1B   | 16     | 2048   | 8192 | 512                   |

Notable: **546M and 1B share the same 16 layers**. Width grows; depth doesn't.
That breaks the simplest "fraction-of-depth" transfer between 546M and 1B —
for the layer-targeting methods (RMU, LUNAR) the same absolute layer index
transfers between 546M and 1B with no rescale; the only depth re-mapping is
12-layer → 16-layer (179M → 546M/1B).

## Common eval-bundle (close-the-loop signal)

Same per-checkpoint triplet across all four methods, mirroring the
gradient-ascent sweep wrapper (`internal/uwiki/eval_ga_sweep_179M.sh`):

- **Forget signal**: `insertion_likelihood --experiment <forget-split>` — CE
  on the forget set itself. Lower is more forgotten.
- **Held-out memorization**: `insertion_likelihood --experiment
  benchmark-contamination-12x` (or another non-target split) — collateral
  damage on memorized content we want preserved.
- **General utility**: `perplexity` on
  `resources/validation-set/c4_en_validation.jsonl` (2500 samples). Per-sample
  JSONL outputs let us bound the regression vs. the baseline at the
  per-checkpoint level.

Anchors to compare against (see `CLAUDE.md` for the full URL list):

- baseline at the loaded step (e.g. `stage1-step100000-tokens210B`),
- continued-pretraining baseline at matching token budget (Exp-Unlearning
  branches `step10{1,...,10}000`),
- ground-truth "deep ignorance" (`OLMo-2-*-Unlearning` repos at the same steps).

Winner criterion at each model size: pick the cell that **minimizes the
forget-side metric subject to a retain-side regression cap** (e.g.
`Δppl_c4 ≤ 5%` vs. baseline). Don't pick on forget alone — every method here
has a configuration that flattens the forget metric while wrecking the model.

## Method 1 — Gradient Ascent (`gradient_ascent.py`)

### 179M anchor sweep (already in flight)

Phase-1 grid is **LR × micro-batch = {1e-6, 5e-6} × {16, 32, 64}**, 20 epochs,
checkpoints at `{5, 10, 15, 20}`. Forget split: single-experiment whitelist
for the in-flight sweep (`memorization-patterns-rare-1-token-1x`); going
forward use the library default (full minus `iid-replacements-*`).

Why this grid (not Jang's 5e-5): the "decade-lower" caveat already documented
in the GA section above — our continued-pretraining horizon dwarfs Jang's
~hundreds of optimizer steps, so the same nominal LR diverges much harder.

### 179M → 546M / 1B transfer

For Adam-style fine-tuning the consensus rule is **LR ≈ constant across model
sizes** when batch size and sequence length are held fixed. Jang reports
"larger models forget faster at the same LR" (their Table 3). So:

- **Transfer rule**: keep the 179M winner LR; run a small ±0.5-decade
  tightening sweep on the larger models.
- **Epochs**: scale *down* by Jang's empirical ratio (5.4 / 11 ≈ 0.5 from
  125M → 2.7B); for our 179M → 1B step, expect ~ 0.6× the epochs to reach the
  same forget metric. Use that to pre-emptively reduce `--epochs` by ~30 % at
  1B if it speeds the sweep, but a longer cap is harmless because the
  20-epoch budget is hard-capped already.
- **Batch**: keep the 179M winner micro-batch unchanged; gradient memory at
  larger sizes may force `--gradient-accumulation-steps > 1` to preserve the
  effective batch.

### 546M / 1B confirmation sweep

| Cell                | Justification                                 |
| ------------------- | --------------------------------------------- |
| `LR_winner_179M`    | direct transfer                               |
| `0.3 · LR_winner`   | guard against larger-model gradient amplification |
| `3.0 · LR_winner`   | larger models forget faster — may want to push |

Total: 3 cells × 1 micro-batch (the 179M winner) at each scale. Re-run the
eval-bundle on every saved checkpoint.

## Method 2 — SimNPO (`simnpo.py`)

### 179M anchor sweep

SimNPO has three knobs that interact: **β** (saturation temperature),
**γ** (margin), and **LR**. Anchor at **`α_retain = 1.0`** for all cells —
varying α confounds the β/γ search.

| Stage  | Sweep                                                            | Cells | Notes |
| ------ | ---------------------------------------------------------------- | ----- | ----- |
| **A**  | β ∈ {0.1, 0.5, 1.0, 2.5}, γ = 0, LR = 1e-5                       | 4     | Locate the saturation regime that matches our forget set's NLL distribution. Lower β → stronger pressure but earlier collapse. |
| **B**  | γ ∈ {0, 0.5, 1.0, 2.0} at β_winner_A, LR = 1e-5                  | 4     | γ tightens the unlearning target once β is calibrated. |
| **C**  | LR ∈ {5e-6, 1e-5, 5e-5} at (β, γ)_winner, micro-batch ∈ {4, 8}    | 6     | Final sensitivity. |

14 total cells at 179M, ~1 GPU-hour each → ~ 14 GPU-hours; comfortable in a
single `p_datamining` day on 1× H100.

### 179M → 546M / 1B transfer

β and γ are **loss-shape parameters operating on per-token NLL** (in nats);
they are *not* a function of hidden size or layer count. Expect them to
transfer cleanly across model sizes — sanity-check this on a 1-cell run at
546M before committing to the larger compute.

- **Transfer rule**: **(β, γ)** copy verbatim from 179M winner. **LR** copies
  too (Adam-style, same logic as GA), with the same ±0.5-decade tightening
  sweep at 546M / 1B.
- **`α_retain`** likewise stable; revisit only if the retain-side regression
  exceeds the cap.

### 546M / 1B confirmation sweep

3 cells = LR ∈ {0.5, 1.0, 2.0} × LR_winner_179M at fixed (β, γ).

## Method 3 — RMU (`rmu.py`)

### Knobs and their scaling

RMU has four knobs that fall into three different scaling regimes:

| Knob               | Scales with                              | Anchor at 179M     | Transfer rule                                 |
| ------------------ | ---------------------------------------- | ------------------ | --------------------------------------------- |
| `--target-layer` ℓ | depth fraction                           | ℓ ∈ {3, 5, 7}      | 179M ℓ × 16/12 → nearest int (same for 546M and 1B) |
| `--steering-coef` c | √hidden_size *(approx; verify by measuring `‖h_ℓ_frozen‖`)* | c ∈ {2, 4, 6}      | c_546M ≈ c_179M · √(1120/576) ≈ 1.39·c; c_1B ≈ c_179M · √(2048/576) ≈ 1.89·c |
| `--alpha` α        | ratio of forget-MSE / retain-MSE         | α ∈ {600, 1200}    | constant (both MSEs scale ≈ proportionally with hidden) |
| `--learning-rate`  | Adam fine-tuning, ≈ constant             | LR = 5e-5          | constant; ±0.5-decade tightening              |

`--n-layers-to-update = 3` (paper) and `--frozen-dtype bfloat16` are fixed.

### 179M anchor sweep

| Stage | Sweep                                                          | Cells | Notes |
| ----- | -------------------------------------------------------------- | ----- | ----- |
| **A** | ℓ ∈ {3, 5, 7}, c = 4, α = 1200, LR = 5e-5                       | 3     | Locate the depth where redirection is most effective without breaking the residual stream. Mid-network is the paper's prior. |
| **B** | c ∈ {2, 4, 6, 8} at ℓ_winner_A, α = 1200, LR = 5e-5             | 4     | Calibrate steering magnitude to OLMo-2's 179M activation norms. **Before this stage, log `‖h_ℓ_frozen‖₂` for a small retain batch** so c can be re-anchored if our scaling estimate is off. |
| **C** | LR ∈ {1e-5, 5e-5, 1e-4} at (ℓ, c)_winner, α ∈ {600, 1200, 2400} | 9     | Final calibration. |

16 total cells; modest because the per-cell run is short (paper uses ~100–200
optimizer steps). Forget split = library default.

### 179M → 546M / 1B transfer

- **Layer ℓ**: convert depth fraction `ℓ_179M / 12` to absolute index in a
  16-layer model: `round((ℓ_179M / 12) · 16)`. Concretely, ℓ=5 at 179M →
  ℓ=7 at 546M and 1B. Since 546M and 1B both have 16 layers, the layer
  transfers between them with no further change.
- **c**: scale by √(hidden_ratio) — anchor 179M c_winner; 546M ≈ 1.39 ·
  c_winner; 1B ≈ 1.89 · c_winner. Sanity-check by measuring `‖h_ℓ_frozen‖₂`
  on a retain batch at each scale (one print line; one sbatch job each) and
  adjust c so that **c / mean‖h‖** is preserved across scales.
- **α**: constant (1200 default).
- **LR**: constant; tighten ±0.5 decade.

### 546M / 1B confirmation sweep

4 cells = c ∈ {0.7, 1.0, 1.4} × c_predicted (covers the case where the
√hidden rule is off) plus LR ∈ {0.5, 1.0} × LR_winner.

### Caveat for RMU

The paper's c=6.5 and α=1200 are calibrated to **Llama-2-7B layer-7**
post-instruction-tuning activations, which are noticeably more peaked than
pretrain-only OLMo-2 activations. Don't be surprised if our 179M winner has
c in the {2, 4} range and α somewhat below 1200. The **c-via-√hidden**
transfer rule is a *first-pass* approximation; the principled version is to
match `c / mean‖h_ℓ_frozen‖₂` across model sizes, which is one extra eval
job per model.

## Method 4 — LUNAR (`lunar.py`)

### Knobs and their scaling

| Knob                      | Scales with                  | Anchor at 179M  | Transfer rule        |
| ------------------------- | ---------------------------- | --------------- | -------------------- |
| `--redirection-layer` ℓ   | depth fraction               | ℓ ∈ {4, 6, 8}   | 179M ℓ × 16/12 → nearest int |
| `--retain-loss-weight` α  | MSE-ratio (≈ invariant)       | α ∈ {0.5, 1, 2} | constant             |
| `--learning-rate`         | Adam fine-tuning             | LR = 5e-5       | constant; tighten    |
| `--anchor-num-tokens`     | n/a (anchor lives in the *frozen* model) | 1               | constant             |

`--update-scope full-layer`, `--anchor-source eos`, `--frozen-dtype bfloat16`
all fixed by user choice.

### 179M anchor sweep

| Stage | Sweep                                            | Cells |
| ----- | ------------------------------------------------ | ----- |
| **A** | ℓ ∈ {4, 6, 8}, α = 1.0, LR = 5e-5                 | 3     |
| **B** | α ∈ {0.5, 1.0, 2.0} at ℓ_winner, LR = 5e-5        | 3     |
| **C** | LR ∈ {1e-5, 5e-5, 1e-4} at (ℓ, α)_winner          | 3     |

9 cells. The anchor (an EOS-only-derived activation) is one H-dim vector and
has no scale-dependent calibration of its own — it's whatever the frozen
model produces.

### 179M → 546M / 1B transfer

- **Layer ℓ**: same depth-fraction rule as RMU (12-layer → 16-layer).
- **α**: constant. Both forget-MSE (`h − anchor`) and retain-MSE
  (`h − h_frozen`) scale proportionally with hidden size, so their ratio is
  invariant.
- **LR**: constant; tighten ±0.5 decade.

### 546M / 1B confirmation sweep

3 cells = LR ∈ {0.5, 1.0, 2.0} × LR_winner at fixed (ℓ, α).

## Compute envelope (single H100, 1× cell)

| Method  | 179M cell | 546M cell | 1B cell | 179M anchor sweep total |
| ------- | --------- | --------- | ------- | ----------------------- |
| GA      | ~30 min   | ~1.5 h    | ~3 h    | ~3 GPU-h (6-cell grid)   |
| SimNPO  | ~1 h      | ~3 h      | ~6 h    | ~14 GPU-h (14-cell grid) |
| RMU     | ~30 min   | ~1.5 h    | ~3 h    | ~8 GPU-h (16-cell grid)  |
| LUNAR   | ~30 min   | ~1.5 h    | ~3 h    | ~5 GPU-h (9-cell grid)   |

Eval bundle adds ~1.5 GPU-h per checkpoint at 179M, ~4 h at 1B
(insertion_likelihood is the dominant cost). Plan for ~2–3× the training
compute on the eval side.

## Suggested order of work

1. **GA at 179M** — already in flight; pick winner.
2. **SimNPO at 179M** — single sweep run; cheapest tunable method.
3. **RMU at 179M** — instrument the activation-norm logging in stage A so the
   √hidden-size c-transfer can be sanity-checked once.
4. **LUNAR at 179M** — most untested method; expect to revisit anchor
   choice if the EOS analog underperforms.
5. **Confirmation sweeps at 546M / 1B** — only after all four 179M winners
   are in hand, so the cross-method comparison story holds at each scale.
