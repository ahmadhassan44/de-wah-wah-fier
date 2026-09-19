"""Score a de-wah-wah result against its input, without listening.

    uv run python experiments/score.py <input.wav> <result.wav> <segments.txt>

Reports, using pyannote's segments as a rough map:
  - audience-only stretches: level change (want a large drop)
  - poet-alone stretches: level change (want ~0 dB; a drop means her voice is damaged)
  - pitch of what was kept vs removed (for a female poet with a mostly male
    audience: kept should be almost all >165 Hz, removed mostly <150 Hz)
"""
import sys

import librosa
import numpy as np

SR = 16000
inp, res, segs_path = sys.argv[1:4]
x, _ = librosa.load(inp, sr=SR, mono=True)
y, _ = librosa.load(res, sr=SR, mono=True)
n = min(len(x), len(y))
x, y = x[:n], y[:n]

segs = [(float(l.split()[0]), float(l.split()[1]), l.rstrip().endswith("MAIN")) for l in open(segs_path)]
main = np.zeros(n, bool)
other = np.zeros(n, bool)
for a, b, m in segs:
    (main if m else other)[int(a * SR) : int(b * SR)] = True


def db(v):
    return 10 * np.log10(np.mean(v**2) + 1e-12)


def pitch(v):
    f, voiced, _ = librosa.pyin(v, fmin=70, fmax=500, sr=SR, frame_length=1024)
    f = f[voiced & ~np.isnan(f)]
    if not len(f):
        return "no voiced frames"
    return f"median {np.median(f):4.0f} Hz, female-range {np.mean(f > 165) * 100:3.0f}%, male-range {np.mean(f < 150) * 100:3.0f}%"


print("Audience-only stretches (want a big drop):")
drops = []
for a, b, m in segs:
    lo, hi = int(a * SR), int(b * SR)
    sel = ~main[lo:hi]
    if m or sel.sum() < 0.3 * SR:
        continue
    d = db(y[lo:hi][sel]) - db(x[lo:hi][sel])
    drops.append(d)
    print(f"  {int(a // 60)}:{a % 60:05.2f}-{int(b // 60)}:{b % 60:05.2f}  {d:+6.1f} dB")
print(f"  median {np.median(drops):+.1f} dB")

alone = main & ~other
print(f"\nPoet-alone stretches (want ~0 dB): {db(y[alone]) - db(x[alone]):+.1f} dB over {alone.sum() / SR:.0f}s")
print(f"\nKept:    {pitch(y)}")
print(f"Removed: {pitch(x - y)}")

# Cut-offs: moments where the input has the poet (female-range pitch, loud
# enough to be the main voice) but the output is >10 dB quieter there.
HOP = 512
f0, voiced, _ = librosa.pyin(x, fmin=70, fmax=500, sr=SR, frame_length=2048, hop_length=HOP)
lvl_x = librosa.amplitude_to_db(librosa.feature.rms(y=x, frame_length=2048, hop_length=HOP)[0], ref=1.0)
lvl_y = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0], ref=1.0)
k = min(len(f0), len(lvl_x), len(lvl_y))
her = voiced[:k] & (np.nan_to_num(f0[:k]) > 165) & (lvl_x[:k] > np.percentile(lvl_x, 50))
cut = her & (lvl_y[:k] - lvl_x[:k] < -10)
runs, start = [], None
for i, c in enumerate(np.append(cut, False)):
    if c and start is None:
        start = i
    elif not c and start is not None:
        if (i - start) * HOP / SR >= 0.15:
            runs.append((start * HOP / SR, i * HOP / SR))
        start = None
lost = sum(b - a for a, b in runs)
print(f"\nPoet cut off (her pitch in input, output >10 dB quieter): {lost:.1f}s of {her.sum() * HOP / SR:.0f}s of her voice")
for a, b in runs:
    print(f"  {int(a // 60)}:{a % 60:05.2f}-{int(b // 60)}:{b % 60:05.2f}  ({b - a:.2f}s)")
