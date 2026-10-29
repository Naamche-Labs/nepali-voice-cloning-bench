"""Score a system on the Nepali Voice Cloning Benchmark.

    python score.py --data DATA_DIR --audio SYSTEM_DIR --name NAME [--track raw|clean]

DATA_DIR is the Hugging Face dataset (Aarjan/nepali-voice-cloning-bench) downloaded locally.
SYSTEM_DIR holds one file per item, named <id>.wav (any sample rate), each saying that item's
target_text in the voice of its prompt clip. Writes results/NAME.json and prints the three scores:

- CER: character error rate (%) of facebook/mms-1b-all (Nepali adapter, Latin output masked) against
  the target text. Digits are spelled out on both sides; punctuation and spacing are ignored.
- Voice match: ECAPA (speechbrain/spkrec-ecapa-voxceleb) cosine similarity between the output and the
  real recording of the same person saying the same sentence.
- UTMOS: predicted naturalness (UTMOS22 strong, via tarepan/SpeechMOS v1.2.0), 1-5.
"""
import argparse
import json
import os
import re
import unicodedata

import librosa
import torch

from nepali_norm import normalize

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def norm(s):
    if re.search(r"[0-9०-९]", s):
        s = normalize(s, english=False)
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[।॥.,?!;:\"'()\-–—]", " ", s)
    return " ".join(s.split())


def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


class MMS:
    def __init__(self):
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi",
                                                    ignore_mismatched_sizes=True).to(DEV).eval()
        vocab = self.proc.tokenizer.get_vocab()
        self.mask = torch.zeros(len(vocab), device=DEV)
        for tok, i in vocab.items():  # Nepali references are Devanagari: never let the recogniser answer in Latin
            if re.search(r"[A-Za-z]", tok) and not tok.startswith("<"):
                self.mask[i] = float("-inf")

    def __call__(self, w):
        inp = {k: v.to(DEV) for k, v in self.proc(w, sampling_rate=16000, return_tensors="pt").items()}
        with torch.no_grad():
            return self.proc.decode((self.model(**inp).logits[0] + self.mask).argmax(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--track", default="clean", choices=["raw", "clean"])
    ap.add_argument("--ceiling", action="store_true", help="score the real recordings: voice match against the prompt clip")
    a = ap.parse_args()
    from speechbrain.inference.speaker import EncoderClassifier
    asr = MMS()
    spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEV})
    mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()

    def emb(w):
        with torch.no_grad():
            return spk.encode_batch(torch.tensor(w).unsqueeze(0).to(DEV)).squeeze()

    rows = []
    for line in open(os.path.join(a.data, "items.jsonl"), encoding="utf-8"):
        it = json.loads(line)
        p = os.path.join(a.audio, f"{it['id']}.wav")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}: every item must have an output (a failed item counts, it is not skipped)")
        w, _ = librosa.load(p, sr=16000)
        real, _ = librosa.load(os.path.join(a.data, it["prompt_audio" if a.ceiling else "real_audio"]), sr=16000)
        ref = norm(it["target_text"])
        hyp = norm(asr(w)) if len(w) > 1600 else ""
        with torch.no_grad():
            m = float(mos(torch.tensor(w).unsqueeze(0).to(DEV), 16000)) if len(w) > 1600 else 1.0
        sim = float(torch.nn.functional.cosine_similarity(emb(w), emb(real), dim=0)) if len(w) > 1600 else 0.0
        rows.append({"id": it["id"], "errors": ed(ref, hyp), "chars": len(ref), "hyp": hyp, "sim": sim, "utmos": m})
    n = len(rows)
    res = {"name": a.name, "track": a.track, "n": n,
           "cer": round(100 * sum(r["errors"] for r in rows) / sum(r["chars"] for r in rows), 2),
           "voice_match": round(sum(r["sim"] for r in rows) / n, 3),
           "utmos": round(sum(r["utmos"] for r in rows) / n, 2), "rows": rows}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/{a.name}.json", "w"), ensure_ascii=False, indent=1)
    print(f"{a.name} [{a.track}]  CER {res['cer']:.2f}  voice match {res['voice_match']:.3f}  UTMOS {res['utmos']:.2f}  (n={n})")
    print("per item (v1.0). For the v1.1 headline, each distinct prompt + sentence pair counted once, run: python summarize.py")


if __name__ == "__main__":
    main()
