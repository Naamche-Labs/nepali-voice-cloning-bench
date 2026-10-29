# Nepali Voice Cloning Benchmark

An open test of zero-shot voice cloning in Nepali. A system hears one short clip of a person it has never heard, then says a new Nepali sentence in that voice. We score how correctly it speaks, how much it sounds like that person, and how natural it sounds.

- **Audio:** [huggingface.co/datasets/Aarjan/nepali-voice-cloning-bench](https://huggingface.co/datasets/Aarjan/nepali-voice-cloning-bench)
- **Code and leaderboard:** this repository
- **Built by:** [Naamche Labs](https://github.com/Naamche-Labs), alongside [Swar](https://github.com/Naamche-Labs/swar)

## Leaderboard

### Clean track (prompts denoised, the main track)

| # | System | CER ↓ | Voice match ↑ | UTMOS ↑ |
|---|---|---|---|---|
| 1 | [Swar](https://github.com/Naamche-Labs/swar) (Naamche Labs) | **4.72** | **0.697** | **2.71** |
| 2 | [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (k2-fsa) | 5.08 | 0.683 | 2.62 |
| 3 | [IndicF5](https://huggingface.co/ai4bharat/IndicF5) (AI4Bharat) | 8.30 | 0.669 | 2.67 |
| | *Real recordings (ceiling)* | *10.30* | *0.742* | *2.32* |

### Raw track (prompts as recorded)

| # | System | CER ↓ | Voice match ↑ | UTMOS ↑ |
|---|---|---|---|---|
| 1 | [Swar](https://github.com/Naamche-Labs/swar) (Naamche Labs) | **4.59** | **0.700** | 2.46 |
| 2 | [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (k2-fsa) | 5.34 | 0.689 | **2.60** |
| 3 | [IndicF5](https://huggingface.co/ai4bharat/IndicF5) (AI4Bharat) | 8.49 | 0.659 | 2.43 |
| | *Real recordings (ceiling)* | *10.30* | *0.742* | *2.32* |

Ranked by CER. Bold is best in each column. v1.1 (2026-10-29): each of the 64 distinct prompt + sentence pairs counts once (see *What changed in v1.1*); the audio and per-item scores are the ones from 2026-10-25. The real recordings' CER is high because the recogniser also struggles with phone-quality speech; their voice match (a person against their own other clip) is the best possible.

Higher voice match and UTMOS are better; lower CER is better. Every system gets exactly the same prompt clips, transcripts and target sentences, and the same scorer (`score.py`). A failed or empty output is scored as fully wrong, never skipped.

## The data

- **Source:** the Nepali (ne_np) test split of [Google FLEURS](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0), read speech recorded in Nepal.
- **82 items, 64 distinct prompt + sentence pairs, 50 distinct sentences, 4 voice groups** (25, 25, 25 and 7 items). FLEURS has several recordings of many sentences, so some pairs appear 2-3 times, each time with a different real recording. Each item has a `pair` id in `items.jsonl`.
- **Voice groups, not verified speakers.** FLEURS has no speaker labels, so test recordings were grouped by ECAPA voice embeddings (average-linkage clustering, cosine distance 0.6, within gender), and one 4-12 s clip of each group is its prompt. Grouping can merge similar voices: in group `0_0`, items 000, 003 and 020 have real recordings about 8 semitones above the group's typical pitch, most likely another reader. They are marked `real_other_reader: true`. Every system is scored against the same recordings, so this affects all systems equally; it does lower the real-recording ceiling for those items.
- **Each item** (`items.jsonl`) has the prompt clip and its transcript, the target sentence, and the real recording of that person saying the target sentence.
- **The real recordings are the ceiling.** Scored the same way, they show the best voice match possible (a person against their own other clip).

## Two tracks

FLEURS clips are recorded on phones and are noisy, and cloning models copy the noise.

- **Raw track:** the prompt clips exactly as FLEURS recorded them (`prompts/`).
- **Clean track:** the same clips denoised with DeepFilterNet3, attenuation capped at 18 dB (`prompts_clean/`). The cap was chosen on FLEURS validation speakers, which are disjoint from these.

Report the track you used. Do not apply other preprocessing to the prompts without saying so.

## Scores

`score.py` scores all 82 items; `summarize.py` then averages within each distinct pair and over the 64 pairs (the headline numbers). The v1.0 per-item averages stay in each result file under `per_item`.

| Score | How | Better |
|---|---|---|
| **CER** | Character error rate (%) of [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) with the Nepali adapter (Latin output masked) against the target text. Digits spelled out on both sides; punctuation and spacing ignored. | Lower |
| **Voice match** | Cosine similarity of [ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) speaker embeddings between the output and the real recording of the same person saying the same sentence. | Higher |
| **UTMOS** | Predicted naturalness, UTMOS22 strong via [SpeechMOS](https://github.com/tarepan/SpeechMOS) v1.2.0, 1-5. | Higher |

All three are automatic. UTMOS was trained on English, and FLEURS audio is phone quality, so absolute UTMOS values are low for everyone; compare systems, not absolute values.

## Pitch check (v1.1)

`pitch_check.py` measures the median pitch (F0) of every prompt, real recording and clean-track output (`results/pitch.json`). A system should keep the pitch of its **prompt**, the voice it is asked to copy. Mean gap to the prompt over all 82 items:

| | Gap to prompt pitch ↓ |
|---|---|
| **Swar** | **11.5 Hz (1.53 semitones)** |
| IndicF5 | 12.7 Hz (1.77 semitones) |
| OmniVoice | 19.6 Hz (2.01 semitones) |
| *Real recordings* | *14.7 Hz (1.74 semitones)* |

Per voice group the picture differs: in group `0_0` Swar and IndicF5 come out lower than the prompt (110 and 105 Hz against 139 Hz), OmniVoice higher (186 Hz). Pitch is not part of the ranking; it is reported because a speaker embedding does not capture it. Thanks to the reviewer in discussion #2 for raising it.

## What changed in v1.1 (2026-10-29)

- Headline scores count each distinct prompt + sentence pair once (64 pairs) instead of each of the 82 items. Rankings are unchanged on both tracks.
- `items.jsonl` gained `pair` and `real_other_reader`.
- Added the pitch check and corrected the data description (items per group, repeated sentences, grouping caveat).

## Submit a system

1. Download the dataset: `huggingface-cli download Aarjan/nepali-voice-cloning-bench --repo-type dataset --local-dir data`
2. For every line of `data/items.jsonl`, clone the voice of `prompt_audio` (raw track) or `prompt_audio_clean` (clean track) using `prompt_text` as its transcript if your system takes one, and say `target_text`. Save it as `<id>.wav`.
3. `pip install -r requirements.txt`, then `python score.py --data data --audio YOUR_DIR --name YOUR_SYSTEM --track raw`.
4. Open a pull request adding `results/YOUR_SYSTEM.json` and a row to the leaderboard, with a link to the model and the exact settings used. Uploading the audio as well (for example to a Hugging Face dataset) lets others check it.

## How the listed systems were run

- **Swar** (Naamche Labs): final model, NFE 16, CFG 2.0, seed 0, one take per item, pace matched to the prompt.
- **OmniVoice** (k2-fsa/OmniVoice, 0.6B): default settings, `language="npi"`, prompt transcript given.
- **IndicF5** (ai4bharat/IndicF5), unchanged: NFE 16, CFG 2.0, seed 0, the same pacing as Swar.

Outputs of every listed system are in the dataset under `outputs/<track>/<system>/`.

## Licence

Code: MIT. Audio and text derived from FLEURS: CC BY 4.0 (Google). System outputs: released under CC BY 4.0 by Naamche Labs for checking the results.

## Citation

```bibtex
@misc{nepalivcbench2026,
  title  = {Nepali Voice Cloning Benchmark},
  author = {Chaudhary, Aarjan},
  year   = {2026},
  note   = {Naamche Labs. https://github.com/Naamche-Labs/nepali-voice-cloning-bench}
}
```

FLEURS: Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", SLT 2022.
