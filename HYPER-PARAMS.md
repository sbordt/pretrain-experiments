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
