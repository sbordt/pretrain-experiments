"""Summarize Gaussian Watermark scores for a mid-training (stage2) eval tree.

Usage: python gw_summary_mid.py <EVAL_ROOT>
where EVAL_ROOT = evals/gn-eval3-sweep/<SIZE>-Mid

For each (family, step) it reports mean +/- sem of the in-distribution scores
(dot products with the real injected noise) and the fresh-noise control, plus
signal = mean_in / sem_in (how many SEM the in-dist mean sits from zero).

Validation logic: exp-mid (saw canaries) should show a clear signal
(|mean_in/sem_in| >> 3); deep-ignorance (never saw them) should be ~null.
"""
import glob
import os
import sys

import torch

BASE = sys.argv[1]


def step_key(d):
    return int(os.path.basename(d).replace("step-", ""))


def load(kind, gwdir):
    hits = glob.glob(os.path.join(gwdir, f"gaussian_privacy_scores_{kind}_*.pt"))
    if not hits:
        return None
    return torch.load(sorted(hits)[0], map_location="cpu").float().flatten()


def stats(t):
    n = t.numel(); m = t.mean().item(); s = t.std().item()
    return n, m, s, (s / (n ** 0.5) if n else float("nan"))


hdr = f"{'family':<18} {'step':>6} {'n':>4} {'mean_in':>10} {'sem_in':>8} {'mean_out':>10} {'sem_out':>8} {'signal=in/sem':>14}"
for family in sorted(os.listdir(BASE)):
    fdir = os.path.join(BASE, family)
    if not os.path.isdir(fdir):
        continue
    print(f"\n### {family} ###")
    print(hdr); print("-" * len(hdr))
    for sdir in sorted(glob.glob(os.path.join(fdir, "step-*")), key=step_key):
        gw = os.path.join(sdir, "gaussian_watermark")
        din, dout = load("in", gw), load("out", gw)
        if din is None or dout is None:
            print(f"{family:<18} {step_key(sdir):>6}   MISSING .pt"); continue
        n, m_in, _, sem_in = stats(din)
        _, m_out, _, sem_out = stats(dout)
        sig = m_in / sem_in if sem_in else float("nan")
        print(f"{family:<18} {step_key(sdir):>6} {n:>4} {m_in:>10.4f} {sem_in:>8.4f} "
              f"{m_out:>10.4f} {sem_out:>8.4f} {sig:>14.1f}")
