# Gaussian (Gradient) Noise Unlearning

Training-time unlearning via Gaussian noise on gradient updates. This method is
**no longer the main focus** of the project (which has shifted to evaluating
general unlearning algorithms — see `CLAUDE.md` and `UNLEARNING_EVALUATIONS.md`);
this file preserves the method documentation and the full sweep history.
Model locations (baseline / unlearning-baseline / deep-ignorance, stage1 and
mid-train) are documented centrally in `CLAUDE.md` → "HuggingFace model zoo".

## Gradient Noise experiment type

Added `gradient-noise` experiment type that adds Gaussian noise to all gradient
updates during training. Unlike `gaussian-poisoning` (which targets specific
batches and saves noise vectors), this is global and saves nothing to disk —
only `noise_std` and `seed` are stored.

Config: `type: gradient-noise`, `noise_std: 1e-8`, `seed: 42`.
Env var: `OLMO_GRADIENT_NOISE_CONFIG_FILE`.
Hook module: `pretrain_experiments/gradient_noise.py`.

## Noise-scale scaling analysis

Noise scales are transferred across model sizes via σ ∝ 1/√N, anchored at
179M @ `1e-5`:

| Size | Analog noise scale |
|---|---|
| 179M | `1e-5` (anchor) |
| 546M | ~`5e-6` (also swept `7.5e-6`, `1e-7`) |
| 1B | ~`4e-6` (≈`3.5e-6` using actual non-embedding+embedding count ~1.48B) |
| 2.7B | ~`2.5e-6` (√(179/2700)·1e-5 ≈ 2.6e-6) — **sweep never run** |

## Experiment 1: 179M hyperparameter sweep

Goal: determine the appropriate noise level for the Gaussian unlearning method.

Setup:
- Model: OLMo-2-179M-Exp (small model to save compute)
- Training steps: 10000 (from step 100000), checkpoint every 1000 steps
- Config template: `config/unlearning-gradient-noise-179M.yaml`
- `noise_std` swept: `1e-7, 1e-6, 1e-5` — DONE. Seed: 42.
- Comparisons: baseline (no unlearning, step 100k), unlearning-baseline
  (continued training on remaining data, steps 102k–110k), deep-ignorance
  ground truth (steps 100k–110k). Checkpoint repos/branches: see `CLAUDE.md`.

Evaluations (per checkpoint, every second checkpoint for the noise sweeps):
- **Eval 1 — general capability degradation**: validation loss on 2500 c4
  samples, individual losses saved to file.
  Driver: `internal/uwiki/eval_gn_c4val_sweep.sh` → `evals/gn-c4val-sweep/`.
- **Eval 2 — train-once-answer-all suite**: mathematical reasoning
  (`mathematical_reasoning.py`), denial-of-service (`denial_of_service.py`),
  prompt extraction (`prompt_extraction.py`), benchmark contamination
  (`benchmark.py`). Measure discrete outcome + cross-entropy where possible;
  save per-sample results.
- **Eval 3 — memorization/watermark audit**: fictional_knowledge,
  verbatim_memorization, insertion_likelihood, Gaussian Watermark,
  Memorization Patterns MIA.
  Driver: `internal/uwiki/eval_gn_eval3_sweep.sh` →
  `evals/gn-eval3-sweep/<label>/step-<N>/`. Details (always-on vs. skipped
  evals, `SKIP_IL`, noise-dir conversion, MIA holdout pkl):
  see `UNLEARNING_EVALUATIONS.md` → "Unlearning Eval 3" and `CLAUDE.md`.
  - Checkpoint grid mirrors Eval 1: baseline + unlearning-baseline +
    1e-7/1e-6/1e-5 × {102,104,106,108,110}k + deep-ignorance ×
    {100,102,104,106,108,110}k. The sweep case-block covers
    `MODEL=179M|546M`. `PILOT=1` restricts to baseline + 1e-5@110k.

Hardware/wallclock (training): galadriel (`p_datamining`), 2× H100. Early
attempts on 1× H100 timed out at the 24h budget. Each sweep needed multiple
jobs (sweep + resume) to reach step 110000. Cumulative wallclock:
1e-7 ≈ 1d 22h 23m (2 jobs); 1e-6 ≈ 2d 18h 21m (5 jobs); 1e-5 ≈ 2d 19h 06m
(4 jobs).

Note: once the right noise range is identified on 179M, validate on larger
models (Experiments 2–4).

## Experiment 2: scale-up validation at 546M

Same setup as Experiment 1 but on `OLMo-2-546M-Exp-Unlearning`, continuing from
`step100000-unsharded`. Per the scaling analysis, the 546M analog of
179M @ 1e-5 is around `5e-6`; `1e-7` is deep in the no-effect regime.
Config template: `config/unlearning-gradient-noise-546M.yaml`.

Sweeps:
- `1e-7` (run `dp69aj1f`): complete (steps 101000–110000, checkpoints on disk
  under `unlearning-gradient-noise/OLMo-2-546M-Exp-gradient-noise-dp69aj1f/`)
- `5e-6` (run `ioza65lg`, slurm job 534395): completed 2026-04-25 on
  4× H100 (galadriel)
- `7.5e-6` (slurm job 534699): completed 2026-04-27 on 4× H100 (dgx-h100-em2)
- `1e-6` (run `op80ibt5`): aborted after step 101000-tmp

Hardware (training): 4× H100 (galadriel for `dp69aj1f`/`ioza65lg`, dgx-h100-em2
for `7.5e-6`). Earlier 2× H100 attempts did not finish within the 24h slot.
Cumulative wallclock per sweep: 1e-7 ≈ 4d 23h 54m (across 4 substantive
attempts; multiple cancelled/timeout before 533816 completed in 1d 01h 45m);
5e-6 ≈ 1d 22h 21m (single job 534395); 7.5e-6 ≈ 1d 17h 02m (single job 534699).

## Experiment 3: scale-up validation at 1B

Same setup on `OLMo-2-1B-Exp-Unlearning`, continuing from `step100000-unsharded`.
Model config: `OLMo/configs/official-0425/OLMo2-1B-stage1.yaml` (d_model=2048,
n_layers=16, n_heads=16). Scaling-analysis analog: ~`4e-6` (~3.5e-6 with actual
parameter count). No `config/unlearning-gradient-noise-1B.yaml` template exists.

Notes:
- Training horizon extends to step 150000 / 315B tokens (vs. 110000 / 231B on
  179M and 546M).
- Hardware (training): galadriel, 4× H100. The `3.5e-6` sweep (run `9050f9m3`,
  slurm job 534697) trained steps 100000 → 110000 in **2d 10h 42m**
  (2026-04-25 19:05:56 → 2026-04-28 05:47:47), single job, no resume. Pace
  ~5h 49m per 1k steps in steady state; first 1k included ~20m startup overhead.
- Deep-ignorance per-1k checkpoints mirror the Exp-Unlearning grid 1:1 over
  101k–110k; past step 110000 comparisons are restricted to the every-10k grid
  (see `CLAUDE.md` model zoo for exact branch availability).

## Experiment 4: scale-up validation at 2.7B (not run)

Same setup on `OLMo-2-2.7B-Exp-Unlearning`, continuing from
`step100000-unsharded`. Scaling-analysis analog: ~`2.5e-6`. No
`config/unlearning-gradient-noise-2.7B.yaml` template exists.

Status: the HF checkpoints (baseline / unlearning-baseline / deep-ignorance)
are published, but the gradient-noise sweep has **not** been run — no run IDs,
slurm jobs, or wallclock figures to report. Consequently the 2.7B eval drivers
(`eval_gn_eval3_2.7B_gw.sh`) cover only baseline / unlearning-baseline /
deep-ignorance, with no GN family.

## Other Gaussian noise variants (ideas, not implemented)

- Relative noise scaling: scale noise proportional to gradient magnitude
  (`noise * |grad|`) instead of fixed std, to keep perturbation proportional
  across layers
- Per-layer noise: different noise levels for embeddings, attention, FFN layers
  (gradients vary by orders of magnitude across layers)
- Noise scheduling: start with larger noise and decay, or ramp up over training

## Evaluation and diagnostics ideas

- Track per-step gradient norms alongside noise magnitude to understand
  effective signal-to-noise ratio
- Measure unlearning vs. general degradation tradeoff curves across noise levels
- Compare unlearning durability: does the model re-learn after further clean
  training?
