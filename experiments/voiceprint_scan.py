"""Find stretches of a voice track that do not sound like the main speaker.

After separation + clean-up, audience voices ("wah wah", "mukarrar") survive
because they are speech. This scans the track in short windows, takes an ECAPA
voiceprint of each, learns the main speaker's voiceprint from the windows that
agree with each other most, and flags the rest.

    uv run python experiments/voiceprint_scan.py <voice.wav> <out_dir>

Writes <out_dir>/flagged.wav (only the flagged stretches, for listening) and
<out_dir>/gated.wav (the track with flagged stretches muted), and prints a
timeline.
"""
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier

SR = 16000
WIN, HOP = 1.5, 0.5  # seconds
SPEECH_BELOW = 30.0  # a window is speech if within this many dB of the speaking level

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
x, _ = librosa.load(src, sr=SR, mono=True)

w, h = int(WIN * SR), int(HOP * SR)
starts = np.arange(0, len(x) - w, h)
rms_db = np.array([20 * np.log10(np.sqrt(np.mean(x[s : s + w] ** 2)) + 1e-12) for s in starts])
level = np.percentile(rms_db[rms_db > -80], 75)
speech = rms_db > level - SPEECH_BELOW

enc = EncoderClassifier.from_hparams(
    "speechbrain/spkrec-ecapa-voxceleb", savedir="work/models/ecapa", run_opts={"device": "mps"}
)
idx = np.flatnonzero(speech)
batch = torch.tensor(np.stack([x[starts[i] : starts[i] + w] for i in idx]))
with torch.no_grad():
    emb = torch.cat([enc.encode_batch(b).squeeze(1) for b in batch.split(64)]).cpu().numpy()
emb /= np.linalg.norm(emb, axis=1, keepdims=True)

# Main speaker = the voiceprint most windows agree with: start from the mean,
# then repeatedly re-centre on the 60% of windows closest to it.
centre = emb.mean(0)
for _ in range(5):
    sim = emb @ (centre / np.linalg.norm(centre))
    centre = emb[sim >= np.percentile(sim, 40)].mean(0)
sim = emb @ (centre / np.linalg.norm(centre))

# Threshold: well below the main speaker's typical similarity.
main = sim[sim >= np.percentile(sim, 40)]
thresh = main.mean() - 3 * main.std()
flag = np.zeros(len(starts), bool)
flag[idx[sim < thresh]] = True

# Per-sample mask from flagged windows (a sample is muted if a flagged window covers it
# and no confident main-speaker window does).
bad = np.zeros(len(x), bool)
good = np.zeros(len(x), bool)
for k, i in enumerate(idx):
    s = starts[i]
    (bad if flag[i] else good)[s + h : s + w - h] = True
mute = bad & ~good

print(f"speaking level {level:.1f} dBFS, {len(idx)} speech windows, similarity threshold {thresh:.2f}")
print(f"main-speaker similarity: median {np.median(sim):.2f}, min {sim.min():.2f}")
print(f"flagged {flag.sum()} windows = {mute.sum() / SR:.1f}s of {len(x) / SR:.0f}s\n")
runs, t0 = [], None
for n, m in enumerate(np.append(mute[::SR // 10], False)):
    if m and t0 is None:
        t0 = n / 10
    elif not m and t0 is not None:
        runs.append((t0, n / 10))
        t0 = None
for a, b in runs:
    print(f"  {int(a // 60)}:{a % 60:04.1f} - {int(b // 60)}:{b % 60:04.1f}  ({b - a:.1f}s)")

fade = np.convolve(mute.astype(float), np.ones(480) / 480, mode="same")  # 30 ms soft edges
sf.write(out / "gated.wav", x * (1 - fade), SR)
sf.write(out / "flagged.wav", np.concatenate([x[int(a * SR) : int(b * SR)] for a, b in runs] or [x[:0]]), SR)
