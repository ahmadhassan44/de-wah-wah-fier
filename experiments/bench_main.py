"""Benchmark: can each pipeline model load and run on this Mac (MPS), and how fast?

Usage: uv run python experiments/bench_main.py work/clip.wav
"""
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

clip = Path(sys.argv[1])
out = Path("work/out")
out.mkdir(parents=True, exist_ok=True)
dur = sf.info(clip).duration
results = []


def bench(name, fn):
    t = time.time()
    try:
        info = fn()
        dt = time.time() - t
        results.append((name, "OK", dt, info))
    except Exception as e:  # keep going so we see every model's status
        results.append((name, f"FAIL: {type(e).__name__}: {e}"[:200], time.time() - t, ""))


def run_bs_roformer():
    from audio_separator.separator import Separator

    sep = Separator(output_dir=str(out), output_format="WAV")
    sep.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    files = sep.separate(str(clip))
    return f"device={sep.torch_device} -> {files}"


def run_demucs():
    import demucs.separate

    demucs.separate.main(["-n", "htdemucs", "--two-stems", "vocals", "-d", "mps", "-o", str(out), str(clip)])
    return "htdemucs vocals"


def run_ecapa():
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb", savedir="work/models/ecapa", run_opts={"device": "mps"}
    )
    import librosa

    wav, _ = librosa.load(clip, sr=16000, mono=True)
    x = torch.tensor(wav[: 16000 * 10]).unsqueeze(0)
    emb = enc.encode_batch(x)
    return f"embedding shape={tuple(emb.shape)}"


bench("BS-RoFormer (vocal/background split)", run_bs_roformer)
bench("Demucs htdemucs (vocal/background split)", run_demucs)
bench("ECAPA speaker embedding (voiceprint)", run_ecapa)

print(f"\nClip length: {dur:.0f}s\n")
for name, status, dt, info in results:
    rtf = f"{dur / dt:5.1f}x realtime" if status == "OK" else ""
    print(f"{name:45s} {status:6.200s} {dt:7.1f}s  {rtf}  {info}")
