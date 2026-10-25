# Nepali Voice Cloning Benchmark

An open test of zero-shot voice cloning in Nepali. A system hears one short clip of a person it has never heard, then says a new Nepali sentence in that voice. We score how correctly it speaks, how much it sounds like that person, and how natural it sounds.

- **Audio:** [huggingface.co/datasets/Aarjan/nepali-voice-cloning-bench](https://huggingface.co/datasets/Aarjan/nepali-voice-cloning-bench)
- **Code and leaderboard:** this repository
- **Built by:** [Naamche Labs](https://github.com/Naamche-Labs), alongside [Swar](https://github.com/Naamche-Labs/swar)

## Leaderboard

### Clean track (prompts denoised, the main track)

| # | System | CER ↓ | Voice match ↑ | UTMOS ↑ |
|---|---|---|---|---|
| 1 | [Swar](https://github.com/Naamche-Labs/swar) (Naamche Labs) | **4.55** | **0.691** | **2.67** |
| 2 | [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (k2-fsa) | 4.89 | 0.676 | 2.53 |
| 3 | [IndicF5](https://huggingface.co/ai4bharat/IndicF5) (AI4Bharat) | 8.44 | 0.663 | 2.63 |
| | *Real recordings (ceiling)* | *9.77* | *0.740* | *2.24* |

### Raw track (prompts as recorded)

| # | System | CER ↓ | Voice match ↑ | UTMOS ↑ |
|---|---|---|---|---|
| 1 | [Swar](https://github.com/Naamche-Labs/swar) (Naamche Labs) | **4.58** | **0.694** | 2.39 |
| 2 | [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (k2-fsa) | 5.28 | 0.681 | **2.49** |
| 3 | [IndicF5](https://huggingface.co/ai4bharat/IndicF5) (AI4Bharat) | 8.59 | 0.651 | 2.35 |
| | *Real recordings (ceiling)* | *9.77* | *0.740* | *2.24* |

Ranked by CER. Bold is best in each column. Scored 2026-10-25. The real recordings' CER is high because the recogniser also struggles with phone-quality speech; their voice match (a person against their own other clip) is the best possible.

Higher voice match and UTMOS are better; lower CER is better. Every system gets exactly the same prompt clips, transcripts and target sentences, and the same scorer (`score.py`). A failed or empty output is scored as fully wrong, never skipped.

## The data

- **Source:** the Nepali (ne_np) test split of [Google FLEURS](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0), read speech recorded in Nepal.
- **82 items from 4 speakers.** FLEURS has no speaker labels, so test recordings were grouped into speakers by ECAPA voice embeddings (average-linkage clustering, cosine distance 0.6, within gender). For each speaker, one 4-12 s clip is the prompt and up to 8 of their other sentences are targets.
- **Each item** (`items.jsonl`) has the prompt clip and its transcript, the target sentence, and the real recording of that person saying the target sentence.
- **The real recordings are the ceiling.** Scored the same way, they show the best voice match possible (a person against their own other clip).

## Two tracks

FLEURS clips are recorded on phones and are noisy, and cloning models copy the noise.

- **Raw track:** the prompt clips exactly as FLEURS recorded them (`prompts/`).
- **Clean track:** the same clips denoised with DeepFilterNet3, attenuation capped at 18 dB (`prompts_clean/`). The cap was chosen on FLEURS validation speakers, which are disjoint from these.

Report the track you used. Do not apply other preprocessing to the prompts without saying so.

## Scores

`score.py` computes three numbers over all 82 items:

| Score | How | Better |
|---|---|---|
| **CER** | Character error rate (%) of [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) with the Nepali adapter (Latin output masked) against the target text. Digits spelled out on both sides; punctuation and spacing ignored. | Lower |
| **Voice match** | Cosine similarity of [ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) speaker embeddings between the output and the real recording of the same person saying the same sentence. | Higher |
| **UTMOS** | Predicted naturalness, UTMOS22 strong via [SpeechMOS](https://github.com/tarepan/SpeechMOS) v1.2.0, 1-5. | Higher |

All three are automatic. UTMOS was trained on English, and FLEURS audio is phone quality, so absolute UTMOS values are low for everyone; compare systems, not absolute values.

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
