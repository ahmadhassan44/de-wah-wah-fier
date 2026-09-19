"""Remove audience voices without cutting off the poet.

Muting whole stretches where the audience speaks also mutes the poet whenever
the two overlap. Instead, in those stretches only, swap the mixture for the
poet's stream from a 2-voice separation (MossFormer2). Everywhere else the
original audio passes through untouched.

The separator's stream order is not stable over a long file, so the poet's
stream is chosen per window: whichever stream's voiceprint is closer to the
poet's, learned from the stretches pyannote says are the poet alone.

    uv run python experiments/overlap_rescue.py <mix16k.wav> <s1.wav> <s2.wav> <segments.txt> <out_dir> [--keep 0.0]

--keep is how much audience to leave in: 0 removes it, 0.1 is about -20 dB.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier

SR = 16000
PAD = 0.3   # widen each audience stretch so its edges are covered, seconds
WIN = 1.0   # window for choosing the poet's stream, seconds
FADE = 0.05  # crossfade into and out of the separated stream, seconds

p = argparse.ArgumentParser()
p.add_argument("mix"), p.add_argument("s1"), p.add_argument("s2"), p.add_argument("segments"), p.add_argument("out")
p.add_argument("--keep", type=float, default=0.0)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

x = sf.read(a.mix)[0]
s = np.stack([sf.read(a.s1)[0], sf.read(a.s2)[0]])[:, : len(x)]
n = min(len(x), s.shape[1])
x, s = x[:n], s[:, :n]

segs = [(float(l.split()[0]), float(l.split()[1]), l.rstrip().endswith("MAIN")) for l in open(a.segments)]
other = np.zeros(n, bool)
alone = np.zeros(n, bool)
for st, en, main in segs:
    lo, hi = int(st * SR), int(en * SR)
    if main:
        alone[lo:hi] = True
    else:
        other[max(0, int((st - PAD) * SR)) : int((en + PAD) * SR)] = True
alone &= ~other

enc = EncoderClassifier.from_hparams(
    "speechbrain/spkrec-ecapa-voxceleb", savedir="work/models/ecapa", run_opts={"device": "mps"}
)


def embed(chunks):
    with torch.no_grad():
        e = torch.cat([enc.encode_batch(torch.tensor(np.stack(c))).squeeze(1) for c in np.array_split(chunks, max(1, len(chunks) // 64))])
    e = e.cpu().numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)


# The poet's voiceprint, from 1.5 s pieces of her speaking alone.
w = int(1.5 * SR)
enroll = [x[i : i + w] for i in range(0, n - w, w) if alone[i : i + w].all()]
poet = embed(np.stack(enroll).astype(np.float32)).mean(0)
poet /= np.linalg.norm(poet)
print(f"poet voiceprint from {len(enroll)} x 1.5 s pieces")

# Choose the poet's stream window by window inside the audience stretches.
w = int(WIN * SR)
starts = [i for i in range(0, n - w, w) if other[i : i + w].any()]
choice = np.zeros(n, int)
sims = np.stack([embed(np.stack([s[k, i : i + w] for i in starts]).astype(np.float32)) @ poet for k in (0, 1)])
pick = sims.argmax(0)
for i, c in zip(starts, pick):
    choice[i : i + w] = c
print(f"{len(starts)} windows in audience stretches; stream 1 chosen {np.mean(pick == 0) * 100:.0f}%, "
      f"poet similarity chosen {np.median(sims.max(0)):.2f} vs other {np.median(sims.min(0)):.2f}")

# The separator's streams come out at an arbitrary scale; their sum should
# reproduce the mixture, so fit that one gain (least squares) over the file.
tot = s.sum(0)
gain = float(np.dot(x, tot) / (np.dot(tot, tot) + 1e-12))
her = np.where(choice == 0, s[0], s[1]) * gain
her_plus = her + a.keep * (x - her)

f = int(FADE * SR)
m = np.convolve(other.astype(float), np.ones(f) / f, mode="same")
y = x * (1 - m) + her_plus * m
removed = x - y

print(f"separated stream used for {other.sum() / SR:.1f}s of {n / SR:.0f}s, level match x{gain:.2f}")
sf.write(out / "result.wav", y, SR)
sf.write(out / "removed.wav", removed, SR)
