"""Benchmark ClearerVoice (MossFormer2) models. Runs in envs/clearvoice.

Usage: cd envs/clearvoice && uv run python ../../experiments/bench_clearvoice.py ../../work/clip.wav
"""
import sys
import time
from pathlib import Path

import soundfile as sf
from clearvoice import ClearVoice

clip = Path(sys.argv[1]).resolve()
out = (Path(__file__).parent.parent / "work/out").resolve()
out.mkdir(parents=True, exist_ok=True)
dur = sf.info(clip).duration

for task, model in [
    ("speech_enhancement", "MossFormer2_SE_48K"),  # strips applause / hall noise
    ("speech_separation", "MossFormer2_SS_16K"),  # splits overlapping voices
]:
    t = time.time()
    try:
        cv = ClearVoice(task=task, model_names=[model])
        res = cv(input_path=str(clip), online_write=False)
        cv.write(res, output_path=str(out / f"{model}.wav"))
        dt = time.time() - t
        print(f"{model:22s} OK   {dt:6.1f}s  {dur / dt:5.1f}x realtime")
    except Exception as e:
        print(f"{model:22s} FAIL {type(e).__name__}: {e}"[:300])
