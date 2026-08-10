"""Summarize 2.7B Gaussian Watermark scores: in-distribution (real training
noise) vs out-of-distribution (fresh Gaussian sanity noise) per checkpoint.

For each checkpoint the eval saved:
  gaussian_privacy_scores_in_<step>.pt   -> dot products with the noise that
                                            was actually added during training
  gaussian_privacy_scores_out_<step>.pt  -> dot products with fresh noise
                                            (should sit near 0; sanity control)

A detectable watermark => in-dist mean well above the out-dist mean (in units
of the out-dist spread). We report mean +/- sem for both, and a z-style
separation = (mean_in - mean_out) / std_out.
"""
import glob
import os

import torch

BASE = os.path.expanduser("~/pretrain-experiments/evals/gn-eval3-sweep/2.7B")

# (family, step) display order: baseline, then unlearning-baseline, then deep-ignorance
ORDER = (
    [("baseline", 100000)]
    + [("unlearning-baseline", s) for s in (102000, 104000, 106000, 108000, 110000)]
    + [("deep-ignorance", s) for s in (100000, 102000, 104000, 106000, 108000, 110000)]
)


def load_scores(family, step):
    d = os.path.join(BASE, family, f"step-{step}", "gaussian_watermark")
    def _one(kind):
        hits = glob.glob(os.path.join(d, f"gaussian_privacy_scores_{kind}_*.pt"))
        if not hits:
            return None
        return torch.load(sorted(hits)[0], map_location="cpu").float().flatten()
    return _one("in"), _one("out")


def stats(t):
    n = t.numel()
    m = t.mean().item()
    s = t.std().item()
    sem = s / (n ** 0.5)
    return n, m, s, sem


hdr = f"{'family':<20} {'step':>7} {'n':>5} {'mean_in':>11} {'sem_in':>9} {'mean_out':>11} {'sem_out':>9} {'sep=(in-out)/std_out':>21}"
print(hdr)
print("-" * len(hdr))
for family, step in ORDER:
    din, dout = load_scores(family, step)
    if din is None or dout is None:
        print(f"{family:<20} {step:>7}   MISSING .pt files")
        continue
    n_in, m_in, s_in, sem_in = stats(din)
    n_out, m_out, s_out, sem_out = stats(dout)
    sep = (m_in - m_out) / s_out if s_out > 0 else float("nan")
    print(f"{family:<20} {step:>7} {n_in:>5} {m_in:>11.5f} {sem_in:>9.5f} "
          f"{m_out:>11.5f} {sem_out:>9.5f} {sep:>21.2f}")
