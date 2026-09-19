# de-wah-wah-fier: R&D log

Goal: take a recorded mushaira (poetry recital) and return only the reciting
poet's voice. Remove the audience's *wah wah*, *subhan Allah*, *bohat achay*,
*mukarrar*, clapping and hall noise, and never cut the poet off. It has to
work for any poet (Parveen Shakir, Faiz, Faraz, ...), so it cannot rely on the
poet's gender or pitch. The end product is a free, open-source web service.

Everything below was measured on one 2-minute test clip unless stated:
*Parveen Shakir in Jashn-e-Mahshar*, 2:00–4:00, mono. The poet reads a line, the
audience responds (mostly men, some shouting over her), and she often repeats
the line.

---

## Summary of where things stand

| # | Approach | Audience removed | Poet cut off? | Verdict |
|---|---|---|---|---|
| 0 | Voice/music separation + speech enhancement (from `fix-zia-voice`) | clapping and hall noise only | no | Kept as stage 1; cannot remove voices |
| 1 | ECAPA voiceprint gate (mute windows that don't match the poet) | yes | **badly** (58% of muted audio was her) | Rejected |
| 2 | pyannote diarization, mute non-poet turns | yes | **yes**, wherever audience overlaps her | Rejected |
| 3 | MossFormer2 2-speaker separation in audience turns | **no** (~4 dB) | n/a | Rejected: crowd is not "one speaker" |
| 4 | pyannote, duck (−12 / −20 dB) instead of mute | partly | **yes** | Rejected by listening |
| 5 | SAM Audio large, text + span anchors | yes (median −41 dB) | **yes**: random whole-chunk dropouts | Rejected |
| 6 | #5 + per-chunk retry | yes (median −23 dB) | **yes**: anchors turn SAM into a gate | Rejected by listening |
| 7 | **SAM text-only in audience turns, original elsewhere** | median −24 dB | measured 0.8 s of 51 s | **Current candidate, awaiting listening** |

The core difficulty: audience reactions **overlap** the poet in time. Any
method that decides *which moments to keep* (gating, ducking) must either keep
the audience or cut her. Only true **source separation** (rewriting the audio
within a moment) can keep her and drop them, and most separation models are
trained on clean two-speaker speech, not a hall of people calling out.

---

## Stage 0: noise and clapping (inherited)

From the sibling project `fix-zia-voice` (music removal under Zia Mohyeddin
recitations), whose README documents the models and pitfalls in detail:

1. **Mel-Band RoFormer vocals** (`vocals_mel_band_roformer.ckpt`, via
   `audio-separator`): splits voice from everything else. Removes clapping and
   hall noise. **34 s for the 120 s clip** on an Apple M3 Pro (MPS), including
   model load.
2. **DeepFilterNet3**: cleans residue. **6 s** for the clip.

What it cannot do, as that README states: *"A separator keeps all voices."*
Wah wah is a voice, so it survives. Everything below operates on the output
of this stage.

Also measured: Demucs `htdemucs` runs on MPS (79.8 s, 1.5× real time); only
`htdemucs_ft` hits the MPS channel limit. The RoFormer is faster and cleaner.

---

## Approach 1: ECAPA voiceprint gate (`experiments/voiceprint_scan.py`)

Slide 1.5 s windows (0.5 s hop), take an ECAPA (SpeechBrain,
`spkrec-ecapa-voxceleb`) embedding of each, learn the main speaker as the
centroid most windows agree with, mute windows below `mean − 3σ` similarity.

- Muted 34 s of 120 s.
- Pitch check (see Scorecard): what was **kept** was 97% female-range pitch and
  1% male-range, so the audience was found. What was **muted** was 58%
  female-range, so more than half of what it removed was the poet.
- Listening: "Parveen muted a lot."

**Why it fails:** recitation varies too much (tarannum vs speech, loud vs soft,
hall echo). Independent short windows of the same voice often look like
different speakers.

---

## Approach 2: pyannote diarization gate (`experiments/diarize_gate.py`)

`pyannote/speaker-diarization-community-1` (pyannote.audio 4.0.7, gated, needs
an HF token). Main speaker = most talk time. Mute other speakers' turns except
where pyannote marks overlap with the main speaker.

- 9.1 s for the clip on MPS. Found 2 speakers: 94.2 s (poet) and 20.4 s
  (audience). Muted 19.8 s; pyannote marked only 0.6 s as overlap.
- Muted audio: 31% female-range, 58% male-range. Of 15 muted turns, 11 were
  clearly male (106–155 Hz median) and 4 were female-range (0:28, 0:31, 1:02,
  1:30).
- Listening: those 4 contain **both** her and the audience; she is cut off.
  Other exclamations (*subhan Allah*, *bohat achay*) said over her survive.

**Why it fails:** diarization labels *time*, not sound. When the audience starts
before she finishes a line, the overlap is labelled "audience" and her line
ending is muted with it. Overlapping exclamations are labelled "poet" and kept.

---

## Approach 3: 2-speaker separation (`experiments/overlap_rescue.py`)

ClearerVoice-Studio `MossFormer2_SS_16K` splits the clip into two streams; in
pyannote's audience turns (±0.3 s), swap in the stream whose ECAPA voiceprint
is closer to the poet's.

- 47 s for the clip on MPS.
- Both output streams had the same pitch profile (231 vs 226 Hz median, 87% vs
  88% female-range): the model's stream order is not stable across its
  internal chunks, so streams must be chosen per window.
- Audience-only turns dropped only **about 4 dB** in most places; one got
  7 dB louder.

**Why it fails:** the model is trained to split two clean speakers. A crowd with
reverb is not a second speaker, so it spreads the crowd across both streams.

Bug worth remembering: fitting the stream's level from regions where the
stream picker hadn't run gave a 3.3× gain. Fix: least-squares fit of
`s1 + s2` to the mixture.

---

## Approach 4: ducking instead of muting (`diarize_gate.py --duck N`)

Same pyannote turns, turned down by 12 dB or 20 dB instead of muted.

- Listening: **both cut her off**, poor results. Since her voice is inside those
  turns, lowering them lowers her.

---

## Approach 5: SAM Audio with span anchors (`experiments/sam_extract.py`)

Meta's **SAM Audio** (Dec 2025; `facebook/sam-audio-{small,base,large}`,
gated) is a diffusion transformer that isolates a sound described by a text
prompt, and optionally by time **anchors**: `+` spans (target present) and `−`
spans (target absent). It outputs `target` and `residual`.

Setup: prompt `"A woman reciting poetry"`; `+` = pyannote poet turns that don't
touch an audience turn; `−` = audience turns; 10 s chunks, 1 s crossfade.

- Audience-only turns: median **−41 dB**.
- Listening: no wah wah, but **she is cut off**.
- Diagnosis: whole 10 s chunks came out as silence (−58 to −78 dB) with a
  constant faint tone (~465–765 Hz). Which chunks failed **changed between runs
  with identical settings**. It is a diffusion model, and some random starts
  converge on "target absent". Measured: 18.4 s of her 51 s of clear voice lost.

| chunk | run A (`both`) | run B (`both`) | `+` only | text only |
|---|---|---|---|---|
| 0–9 s | **−57.8** | −0.1 | **−57.9** | −0.1 |
| 18–27 s | −0.7 | **−22.8** | −0.7 | **−22.3** |
| 45–54 s | −0.5 | −0.5 | **−30.3** | −0.5 |
| 99–108 s | **−21.9** | −0.2 | −1.3 | **−18.0** |

(output level minus input level, dB; bold = dropout)

SAM's own remedy is reranking (`reranking_candidates > 1` with a CLAP +
`facebook/sam-audio-judge` ensemble ranker), but it batches candidates and so
multiplies GPU memory, and we were already at the limit (see Infrastructure).

---

## Approach 6: per-chunk verify-and-retry

After each chunk, check the poet's level in 1 s windows where pyannote says she
speaks alone (no audience there, so any drop is a failure); regenerate from a
new seed (up to 4–6 tries) if any window drops more than 6 dB.

- All chunks passed within 4 tries in `+`-only mode. Poet-alone level −0.2 dB;
  the scorecard detector said only 1.0 s lost.
- Listening: **"not better", still cut off, worst at 0:11–0:18.**
- Trace at 100 ms resolution: with anchors, SAM **copies the `+` spans exactly
  (within 0.1 dB, same pitch) and outputs silence everywhere else**. That
  reproduces the pyannote gate: her line starting at 0:12.1 (347–355 Hz) is
  inside an audience turn and silenced.

Two bugs found in the first version of the check:
1. A long poet turn (0:31.7–0:42.8) that touched a 0.2 s audience blip was
   excluded entirely, so a chunk was verified against under 1 s of her voice.
   Fix: build "poet alone" sample by sample (her turns minus audience turns ±
   0.3 s).
2. Averaging over the whole chunk hid a 1.5 s dropout. Fix: judge the worst
   1 s window.

---

## Approach 7 (current): SAM text-only, original elsewhere (`experiments/composite.py`)

Key observation: **without anchors SAM genuinely separates.** At 0:11.8 the input
is dominated by a man (122 Hz); the text-only output has her voice (339 Hz) at
−13 dB, extracted from under him. Her line start at 0:12.1 is kept.

But text-only SAM still sometimes drops her from stretches where she's alone (3
of 14 chunks failed after 6 tries). Where she's alone, though, the original
already contains only her. So:

- **Poet alone:** original audio, untouched (~90 s of 120 s).
- **Each audience turn ±0.3 s:** SAM text-only output (29.3 s of 120 s).
- 50 ms crossfades at the switches.

Scorecard: audience-only turns median **−24.2 dB**; poet-alone **0.0 dB** (it
is the original); detector says 0.8 s of her voice lost (0:09.7, 0:14.2,
1:31.6). Two audience turns are left nearly untouched (0:28, 1:02); these are
the ones where listening said her voice is present.

**Status: awaiting listening.** Known risk: this still depends on pyannote to
decide *where* to use separation. Audience speech that pyannote labels as
"poet" (exclamations over her) is taken from the original and survives.

---

## Scorecard (`experiments/score.py`)

Listening is the ground truth; the scorecard is a fast proxy so each change
doesn't need a full listen. It uses pyannote turns as a rough map and, for this
recording only, pitch (female poet around 200–350 Hz, mostly male audience
around 100–150 Hz):

- level change in audience-only turns (want large negative)
- level change where the poet is alone (want ~0 dB)
- pitch distribution of kept and removed audio
- "cut off": frames where the input has her pitch and is louder than the
  median, but the output is more than 10 dB quieter

**Known blind spot:** the cut-off detector only counts her *loud* frames. It
missed the quiet line starts and endings that listening caught in approach 6.
It said 1.0 s lost when the result was clearly still cut off. Treat it as
necessary but not sufficient. Pitch is a scorecard for this recording only,
never a method: it cannot separate Faiz from a male audience.

Also: SAM *generates* audio (codec + diffusion); its output is time-aligned
but not sample-identical (waveform correlation 0.19 at zero lag). So
`input − target` does not cancel the poet. Use SAM's own `residual` output
instead.

---

## Infrastructure

### Local: Apple M3 Pro, 18 GB
MPS works for PyTorch 2.14, RoFormer, Demucs `htdemucs`, ECAPA, pyannote,
MossFormer2. SAM Audio imports on macOS but the small model download (5.1 GB)
was abandoned on a slow connection.

### AWS: `g5.xlarge` (NVIDIA A10G 24 GB, 22.06 GiB usable; 16 GB RAM; 4 vCPU)
Deep Learning OSS Nvidia Driver AMI, PyTorch 2.13, Ubuntu 26.04, CUDA 13.0.
Models and caches on the NVMe instance store `/opt/dlami/nvme/dww` (**wiped on
instance stop**); code and envs on the root disk; HF token in
`~/.cache/huggingface/token`; 32 GB swapfile on NVMe.

SAM Audio **large, full precision (fp32)**:

| | measured |
|---|---|
| Checkpoint | 14.86 GB (small 5.1 GB, base 7.73 GB) |
| GPU memory after load (vision encoder kept on CPU) | 18.8 GB |
| Peak, 10 s chunks | **21.9 GB allocated / 22.4 GB reserved**, the limit of the card |
| 30 s chunks | out of memory in the audio codec encode |
| Load time | ~350 s: the checkpoint doesn't fit in 16 GB RAM, mmap + swap |
| Speed, one try per chunk | 56 s for 120 s |
| Speed, verify-and-retry | 91 s (`+` anchors) to 164 s (text only) for 120 s |

**Recommendation for production:** `g6e.xlarge` (L40S 48 GB, 32 GB RAM). It
gives headroom for longer chunks or reranking and fixes the load time. 24 GB
works only at 10 s chunks with nothing else on the GPU.

---

## Environments

Three isolated uv environments, because dependencies conflict:

| env | Python | used for |
|---|---|---|
| `/` (root) | 3.11 | pyannote.audio 4.0.7, speechbrain 1.1.1, demucs 4.1.0, audio-separator 0.47.0, torch 2.14 |
| `envs/clearvoice` | 3.11 | ClearerVoice (`clearvoice`), which pins `soundfile==0.12.1` |
| `envs/samaudio` | 3.11 | `sam-audio` from GitHub, `huggingface_hub<1.0` |

Stage 0 used the `fix-zia-voice` environments (`.venv-roformer`, `.venv-dfn`)
directly.

---

## Pitfalls hit

- **PyTorch has no wheels for Python 3.14**; use 3.11.
- `uv init` inside a project adds the new env as a **workspace member** of the
  parent; use `--no-workspace`.
- **ClearerVoice** pins `soundfile==0.12.1` (own env). Its automatic checkpoint
  download can leave an empty folder if interrupted; fetch
  `alibabasglab/MossFormer2_SS_16K` with `hf download`.
- **SAM Audio on macOS:** `decord` (video) has no arm64 wheel and `xformers`
  won't build. Both are only for video prompting: override them away in uv
  and add a stub `xformers` module (`envs/samaudio/stubs`, loaded via a `.pth`
  file) because `perception_models` imports it at module level.
- **SAM Audio needs `huggingface_hub<1.0`**: its `_from_pretrained` requires
  `proxies`/`resume_download`, which 1.x removed.
- **SAM rankers:** the default config builds an ImageBind visual ranker
  (needs ImageBind) and a CLAP + judge text ranker. Pass
  `visual_ranker=None, text_ranker=None` to `from_pretrained` when generating
  one candidate.
- **SAM vision encoder:** with no video, only `vision_encoder.dim` is read;
  keep it on CPU to save GPU memory.
- **pyannote** community-1 and **SAM Audio** are gated: accept the terms on
  Hugging Face; a fine-grained token needs "read access to public gated
  repos".
- `pkill -f pattern` over ssh can match the ssh session's own command line;
  use `pkill -f "[p]attern"`.
- zsh does not word-split `$var` like bash (`set -- $r` broke a loop).
- The AWS DLAMI already mounts instance storage as LVM at `/opt/dlami/nvme`.
  Check with `blkid` before `mkfs`.

---

## Licences

- **SAM Audio:** SAM License (Meta, Nov 2025). Non-exclusive, worldwide,
  royalty-free; use, reproduce, distribute and modify are allowed. Redistribution
  must include the agreement; publications using it must acknowledge it; trade
  control and prohibited-use terms apply. It is not OSI open source, so the
  service's code can be open source but the weights stay under Meta's licence.
- **pyannote community-1:** gated on Hugging Face. Check its licence terms
  before launch.
- **Recordings:** removing the audience does not change who owns a recording.
  The service should process uploads and return them to the uploader rather
  than host a public library, and accept uploads rather than fetch from YouTube
  (which breaks YouTube's terms).

---

## Next

1. Listen to approach 7 (`composite.py`); if still cut off, trace the exact
   timestamps.
2. Stop relying on pyannote to decide where to separate: run text-only SAM over
   the whole recording and use the original only where SAM's output provably
   dropped the poet (verify-and-retry, then fall back).
3. Try SAM's reranking on a 48 GB GPU to reduce text-only dropouts, and the base
   and small models for speed and cost.
4. Run the full pipeline on a complete recording (30 and 83 minutes).
5. Then the web service: upload → queue → GPU worker → download.

## Scripts

| file | purpose |
|---|---|
| `experiments/bench_main.py`, `bench_clearvoice.py` | load-and-run benchmarks on MPS |
| `experiments/voiceprint_scan.py` | approach 1 |
| `experiments/diarize_gate.py` | approaches 2 and 4 (`--duck`); writes pyannote `segments.txt` |
| `experiments/overlap_rescue.py` | approach 3 |
| `experiments/sam_extract.py` | approaches 5–7 (`--modes both,positive,none`, `--tries`, `--chunk`) |
| `experiments/composite.py` | approach 7 |
| `experiments/score.py` | scorecard |
