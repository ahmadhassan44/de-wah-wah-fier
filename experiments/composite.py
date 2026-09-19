"""Combine the original with SAM's text-only extraction.

With span anchors SAM just copies the "+" spans and silences the rest, which
is the same as muting and clips the poet wherever the audience overlaps her.
With a text prompt alone it genuinely separates: it pulls her voice out from
under the audience. But it also sometimes drops her where she speaks alone.

Where she speaks alone the original already contains only her, so use it
there, and use SAM's separation only around and during audience turns:

    uv run python experiments/composite.py <original.wav> <sam_text_only.wav> <segments.txt> <out.wav> [--margin 0.3]
"""
import argparse

import numpy as np
import soundfile as sf
import torch
import torchaudio

FADE = 0.05  # seconds, crossfade at each switch

p = argparse.ArgumentParser()
p.add_argument("original"), p.add_argument("sam"), p.add_argument("segments"), p.add_argument("out")
p.add_argument("--margin", type=float, default=0.3, help="seconds of SAM output kept either side of each audience turn")
a = p.parse_args()

y, sr = sf.read(a.sam, dtype="float32")
x, xsr = sf.read(a.original, dtype="float32")
if x.ndim > 1:
    x = x.mean(1)
if y.ndim > 1:
    y = y.mean(1)
if xsr != sr:
    x = torchaudio.functional.resample(torch.tensor(x), xsr, sr).numpy()
n = min(len(x), len(y))
x, y = x[:n], y[:n]

# 1 where SAM's separation is used: every audience turn plus a margin.
use_sam = np.zeros(n, np.float32)
for line in open(a.segments):
    s, e, *rest = line.split()
    if not line.rstrip().endswith("MAIN"):
        use_sam[max(0, int((float(s) - a.margin) * sr)) : int((float(e) + a.margin) * sr)] = 1.0
f = int(FADE * sr)
use_sam = np.convolve(use_sam, np.ones(f) / f, mode="same")

sf.write(a.out, x * (1 - use_sam) + y * use_sam, sr)
print(f"SAM separation used for {np.mean(use_sam > 0.5) * n / sr:.1f}s of {n / sr:.0f}s; original elsewhere")
