"""Usage: python pitch_check.py DATA_DIR   (needs numpy and ffmpeg; writes results/pitch.json)
Pitch check for the Nepali Voice Cloning Benchmark (2026-10-29), after a comment on HF discussion #2 said Swar sits
15-65 Hz below speaker 0_0's real recordings. Median F0 of every file with one simple tracker (normalised
autocorrelation, 40 ms frames, 10 ms hop, 65-400 Hz, voiced frames only), for the prompt each system was told to copy,
the real recordings, and each system's clean-track outputs. A system is judged against its PROMPT: that is the voice
it is asked to reproduce (FLEURS often has different people read the same sentence, so the "real" recording of an
item is not always the prompt's speaker)."""
import json, subprocess, sys
import numpy as np
from collections import defaultdict

B = sys.argv[1] if len(sys.argv) > 1 else "."
SR = 16000


def load(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def f0_median(x):
    win, hop = int(0.04 * SR), int(0.01 * SR)
    lo, hi = int(SR / 400), int(SR / 65)
    rms_all = np.sqrt(np.mean(x ** 2)) + 1e-9
    out = []
    for i in range(0, len(x) - win, hop):
        f = x[i:i + win] - x[i:i + win].mean()
        if np.sqrt(np.mean(f ** 2)) < 0.3 * rms_all:
            continue
        ac = np.correlate(f, f, "full")[win - 1:]
        ac = ac / (ac[0] + 1e-9)
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.45:                       # voiced, clearly periodic frame
            out.append(SR / k)
    return float(np.median(out)) if len(out) > 10 else float("nan")


items = [json.loads(l) for l in open(f"{B}/items.jsonl")]
cache = {}
def F(p):
    if p not in cache:
        cache[p] = f0_median(load(f"{B}/{p}"))
    return cache[p]

rows = []
for it in items:
    r = {"id": it["id"], "speaker": it["speaker"], "prompt": F(it["prompt_audio_clean"]), "real": F(it["real_audio"])}
    for s in ("swar", "omnivoice", "indicf5"):
        r[s] = F(f"outputs/clean/{s}/{it['id']}.wav")
    rows.append(r)
json.dump(rows, open("results/pitch.json", "w"), indent=1)

by = defaultdict(list)
for r in rows:
    by[r["speaker"]].append(r)
print("speaker | prompt F0 | real recordings (median, range) | Swar | OmniVoice | IndicF5   (median F0, Hz)")
for spk, rs in sorted(by.items()):
    md = lambda k: np.nanmedian([r[k] for r in rs])
    reals = [r["real"] for r in rs]
    print(f"{spk:7} | {md('prompt'):8.1f}  | {md('real'):6.1f} ({np.nanmin(reals):.0f}-{np.nanmax(reals):.0f})"
          f"            | {md('swar'):6.1f} | {md('omnivoice'):6.1f}    | {md('indicf5'):6.1f}")
print("\nmean absolute gap to the PROMPT's F0 (Hz), all 82 items, and the share of the gap in semitones:")
for s in ("swar", "omnivoice", "indicf5", "real"):
    gaps = [abs(r[s] - r["prompt"]) for r in rows if np.isfinite(r[s]) and np.isfinite(r["prompt"])]
    st = [abs(12 * np.log2(r[s] / r["prompt"])) for r in rows if np.isfinite(r[s]) and np.isfinite(r["prompt"])]
    print(f"  {s:10} {np.mean(gaps):6.1f} Hz   {np.mean(st):4.2f} semitones   (n={len(gaps)})")
sp = by["0_0"][:4]
print("\nspeaker 0_0 items 000-003 (the comment's four):")
for r in sp:
    print(f"  {r['id']}  prompt {r['prompt']:.1f}  real {r['real']:.1f}  swar {r['swar']:.1f}  omni {r['omnivoice']:.1f}  indicf5 {r['indicf5']:.1f}")
