"""Keep only the poet with SAM Audio, prompted by text and time spans.

Gating (muting or ducking whole stretches) cuts the poet off whenever the
audience talks over her. SAM Audio instead extracts one sound continuously.
It is told what to keep with a text description and where to find it with
spans from pyannote: "+" where the poet speaks alone, "-" where only the
audience does. The residual (everything else) is also written, so the
audience can be mixed back in a little instead of removed entirely.

Runs in envs/samaudio:
    cd envs/samaudio && uv run python ../../experiments/sam_extract.py \
        <voice.wav> <segments.txt> <out_dir> [--text "A woman reciting poetry"] [--model small] [--modes both,positive,none]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sam_audio import SAMAudio, SAMAudioProcessor

OVERLAP = 1.0  # seconds shared between chunks, crossfaded
MIN_ANCHOR = 0.5  # ignore spans shorter than this
KEEP_DB = 6.0  # a chunk is accepted if no 1 s poet-alone window drops more than this


def level(v):
    return 10 * np.log10(np.mean(v.astype(np.float64) ** 2) + 1e-12)

p = argparse.ArgumentParser()
p.add_argument("voice"), p.add_argument("segments"), p.add_argument("out")
p.add_argument("--text", default="A woman reciting poetry")
p.add_argument("--model", default="small")
p.add_argument("--modes", default="both",
               help="comma list of anchor modes to run with one model load: "
                    "both (+ poet alone, - audience alone), positive (+ only), none (text only)")
p.add_argument("--chunk", type=float, default=10.0, help="seconds per call; GPU memory grows with it")
p.add_argument("--tries", type=int, default=4,
               help="max generations per chunk; SAM sometimes drops the target from a whole chunk, "
                    "so a chunk that loses the poet where she speaks alone is regenerated")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

name = f"facebook/sam-audio-{a.model}"
CHUNK = a.chunk

# Memory-map the checkpoint instead of reading it into RAM: the large model's
# 15 GB checkpoint does not fit in a 16 GB machine's memory alongside the model.
_torch_load = torch.load
torch.load = lambda *args, **kw: _torch_load(*args, **{**kw, "mmap": True})
t_load = time.time()
# The rankers pick the best of several candidates (reranking_candidates > 1);
# we generate one, and the visual ranker needs ImageBind, so leave them out.
model = SAMAudio.from_pretrained(name, visual_ranker=None, text_ranker=None).eval()
# Without a video prompt only vision_encoder.dim is read, so keep it off the GPU.
vision = model.vision_encoder
model.vision_encoder = None
model.to(device)
model.vision_encoder = vision
print(f"loaded {name} in {time.time() - t_load:.0f}s", flush=True)
if device.type == "cuda":
    print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
proc = SAMAudioProcessor.from_pretrained(name)
sr = proc.audio_sampling_rate if hasattr(proc, "audio_sampling_rate") else 48000

wav, in_sr = sf.read(a.voice, dtype="float32")
if wav.ndim > 1:
    wav = wav.mean(1)
x = torchaudio.functional.resample(torch.tensor(wav), in_sr, sr).numpy() if in_sr != sr else wav
dur = len(x) / sr

# Anchors from diarization: "+" poet alone, "-" audience alone. "Poet alone"
# is built sample by sample (her turns minus the audience's, with a margin),
# so one brief audience blip does not disqualify an entire long turn of hers.
segs = [(float(l.split()[0]), float(l.split()[1]), l.rstrip().endswith("MAIN")) for l in open(a.segments)]
MARGIN = 0.3  # seconds kept clear around each audience turn
poet_alone = np.zeros(len(x), bool)
for s, e, m in segs:
    if m:
        poet_alone[int(s * sr) : int(e * sr)] = True
for s, e, m in segs:
    if not m:
        poet_alone[max(0, int((s - MARGIN) * sr)) : int((e + MARGIN) * sr)] = False


def runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [(b / sr, e / sr) for b, e in zip(edges[::2], edges[1::2])]


spans = [("+", s, e) for s, e in runs(poet_alone) if e - s >= MIN_ANCHOR]
spans += [("-", s, e) for s, e, m in segs if not m and e - s >= MIN_ANCHOR]


def worst_window(y, lo):
    """Largest level drop, output vs input, over 1 s windows where she is alone."""
    w = sr
    drops = []
    for i in range(0, len(y) - w // 2, w // 2):
        m = poet_alone[lo + i : lo + i + w][: len(y) - i]
        if m.sum() < w // 2:
            continue
        seg_y, seg_x = y[i : i + w][: len(m)][m], x[lo + i : lo + i + w][: len(m)][m]
        if level(seg_x) < speech_level - 30:  # a pause, nothing to lose
            continue
        drops.append(level(seg_y) - level(seg_x))
    return min(drops) if drops else 0.0


speech_level = np.percentile([level(x[i : i + sr]) for i in range(0, len(x) - sr, sr)], 75)

step = CHUNK - OVERLAP


def run(mode):
    target = np.zeros(len(x), np.float32)
    weight = np.zeros(len(x), np.float32)
    residual = np.zeros(len(x), np.float32)
    use = [sp for sp in spans if mode == "both" or (mode == "positive" and sp[0] == "+")]
    t0 = time.time()
    log = []
    for chunk_no, c0 in enumerate(np.arange(0, max(dur - OVERLAP, 1e-3), step)):
        c1 = min(c0 + CHUNK, dur)
        lo, hi = int(c0 * sr), int(c1 * sr)
        chunk_path = out / "_chunk.wav"
        sf.write(chunk_path, x[lo:hi], sr, subtype="FLOAT")
        anchors = [[sign, max(s, c0) - c0, min(e, c1) - c0] for sign, s, e in use if s < c1 and e > c0]
        anchors = [an for an in anchors if an[2] - an[1] >= MIN_ANCHOR]
        kw = {"anchors": [anchors]} if anchors else {}
        inputs = proc(audios=[str(chunk_path)], descriptions=[a.text], **kw).to(device)
        # Where the poet speaks alone there is no audience to remove, so the
        # output should keep her level in every 1 s window. A big drop in any
        # window means this generation dropped her; retry from another seed.
        best = None
        for attempt in range(a.tries):
            torch.manual_seed(1000 * chunk_no + attempt)
            with torch.inference_mode():
                res = model.separate(inputs, predict_spans=not kw)
            y = res.target[0].float().cpu().numpy().reshape(-1)[: hi - lo]
            r = res.residual[0].float().cpu().numpy().reshape(-1)[: hi - lo]
            kept = worst_window(y, lo)
            if best is None or kept > best[0]:
                best = (kept, y, r)
            if kept > -KEEP_DB:
                break
        kept, y, r = best
        log.append((c0, c1, attempt + 1, kept))
        # Linear crossfade weights over the overlap at each chunk edge.
        wgt = np.ones(len(y), np.float32)
        f = int(OVERLAP * sr)
        if c0 > 0:
            wgt[:f] = np.linspace(0, 1, f)
        if c1 < dur:
            wgt[-f:] = np.minimum(wgt[-f:], np.linspace(1, 0, f))
        target[lo : lo + len(y)] += y * wgt
        residual[lo : lo + len(r)] += r * wgt
        weight[lo : lo + len(y)] += wgt
    target /= np.maximum(weight, 1e-6)
    residual /= np.maximum(weight, 1e-6)
    (out / "_chunk.wav").unlink(missing_ok=True)
    print(f"{mode:9s}: {time.time() - t0:.0f}s for {dur:.0f}s of audio", flush=True)
    for c0, c1, tries, kept in log:
        note = "" if kept > -KEEP_DB else "  <- still lost after all tries"
        print(f"    {c0:6.1f}-{c1:6.1f}s  {tries} tr{'y' if tries == 1 else 'ies'}, worst poet window {kept:+6.1f} dB{note}", flush=True)
    # SAM's own residual: the generated output is not sample-identical to the
    # input, so input minus target would still contain the poet.
    sf.write(out / f"poet-{mode}.wav", target, sr)
    sf.write(out / f"residual-{mode}.wav", residual, sr)


for mode in a.modes.split(","):
    run(mode)
if device.type == "cuda":
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB allocated, "
          f"{torch.cuda.max_memory_reserved() / 1e9:.1f} GB reserved")
