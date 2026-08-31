# Results — Kannada-English Bilingual ASR

Word Error Rate (WER %) after ~1 epoch (33,265 steps) of finetuning, evaluated
on 4 Kannada benchmark testsets. Lower is better.

| Testset | data2vec-AQC (Kannada fine-tuned) | data2vec-AQC (multilingual SSL) | XEUS |
|---|---|---|---|
| FLEURS | **21.40** | 23.97 | 25.02 |
| IndicTTS | **18.06** | 22.70 | 23.63 |
| Kathbath | **17.03** | 21.01 | 24.64 |
| Kathbath-Noisy | **17.87** | 22.20 | 29.70 |

**data2vec-AQC (Kannada CTC fine-tuned)** wins on every testset — expected,
since it's the only encoder among the three that was already fine-tuned on
Kannada speech (via CTC) before this SLAM-LLM stage; the other two are
general-purpose SSL encoders (multilingual pretrained-only / cross-lingual)
seeing Kannada for the first time here.

## Training accuracy trend

All three encoders were evaluated during training on the held-out dev set
(`eval_acc`, teacher-forced next-token accuracy — not the same metric as WER
above, but a useful convergence signal):

| Encoder | Final eval_acc (epoch 1) | Notes |
|---|---|---|
| data2vec-AQC (Kannada fine-tuned) | 88.0% | Sharp "emergence" jump ~step 19-20k (59%→88%) |
| data2vec-AQC (multilingual SSL) | 87.25% | Sharp "emergence" jump ~step 19-20k (59%→87%) |
| XEUS | 87.0% (epoch 1: 86.75%, epoch 2 plateau ~87%) | Emergence jump ~step 11k (55.6%→79.6%), earlier than the other two, after fixing `dataset_config.normalize` and switching to the official `SSLTask.build_model_from_file()` loader (100% of checkpoint params load, vs. ~98.3% with an earlier manual-patch workaround) |
| Wav2Vec2-BERT 2.0 | training in progress | `dataset_config.encoder_frame_offset` fix applied (corrects a systematic off-by-one in reserved audio-token slots caused by its log-mel + stride-2 feature pipeline) |

## Reproducing

```bash
# Decode (per encoder, 4 testsets each)
bash examples/asr_librispeech/scripts/inference_data2vec_gemma3_kannada_4testsets.sh       # data2vec-AQC (Kannada FT)
bash examples/asr_librispeech/scripts/inference_data2vec_gemma3_kannada_ssl_4testsets.sh   # data2vec-AQC (SSL)
bash examples/asr_librispeech/scripts/inference_data2vec_gemma3_kannada_xeus_4testsets.sh  # XEUS

# Score
bash examples/asr_librispeech/scripts/compute_wer_kannada_3encoders.sh
```
