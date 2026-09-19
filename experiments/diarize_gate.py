"""Mute everyone except the main speaker, using pyannote speaker diarization.

pyannote labels who speaks when across the whole recording (not window by
window), and marks where two people speak at once. The main speaker is the
one with the most talk time. Stretches where only other speakers talk are
muted; stretches where the main speaker overlaps someone else are kept for now
(a later stage separates those).

    uv run python experiments/diarize_gate.py <voice.wav> <out_dir> [--duck 15]

--duck N turns audience stretches down by N dB instead of muting them, so
the poet is still heard where she overlaps the audience.

Writes <out_dir>/gated.wav, <out_dir>/muted.wav (only what was removed) and
<out_dir>/segments.txt.
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline

SR = 16000
FADE = 0.03  # seconds, soft edges on each mute

src, out = Path(sys.argv[1]), Path(sys.argv[2])
duck_db = float(sys.argv[sys.argv.index("--duck") + 1]) if "--duck" in sys.argv else None
out.mkdir(parents=True, exist_ok=True)
x, _ = librosa.load(src, sr=SR, mono=True)

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=True)
pipe.to(torch.device("mps"))
t = time.time()
result = pipe({"waveform": torch.tensor(x).unsqueeze(0), "sample_rate": SR})
print(f"diarization: {time.time() - t:.1f}s for {len(x) / SR:.0f}s of audio")
diar = getattr(result, "speaker_diarization", result)

talk = defaultdict(float)
segs = []
for turn, _, spk in diar.itertracks(yield_label=True):
    talk[spk] += turn.duration
    segs.append((turn.start, turn.end, spk))
main = max(talk, key=talk.get)
for spk, d in sorted(talk.items(), key=lambda kv: -kv[1]):
    print(f"  {spk}: {d:6.1f}s{'  <- main speaker' if spk == main else ''}")

# Per-sample masks: where the main speaker talks, and where anyone else does.
is_main = np.zeros(len(x), bool)
is_other = np.zeros(len(x), bool)
for a, b, spk in segs:
    (is_main if spk == main else is_other)[int(a * SR) : int(b * SR)] = True
mute = is_other & ~is_main
overlap = is_other & is_main
print(f"muted {mute.sum() / SR:.1f}s, overlap kept {overlap.sum() / SR:.1f}s")

with open(out / "segments.txt", "w") as f:
    for a, b, spk in segs:
        f.write(f"{a:8.2f} {b:8.2f} {spk}{' MAIN' if spk == main else ''}\n")

n = int(FADE * SR)
soft = np.convolve(mute.astype(float), np.ones(n) / n, mode="same")
depth = 1.0 if duck_db is None else 1 - 10 ** (-duck_db / 20)
sf.write(out / "gated.wav", x * (1 - depth * soft), SR)
sf.write(out / "muted.wav", x[mute], SR)
