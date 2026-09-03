# Results — Kannada-English Bilingual ASR

Word Error Rate (WER %) after ~1 epoch (33,265 steps) of finetuning, evaluated
on 4 Kannada benchmark testsets. Lower is better.

| Testset | data2vec-AQC (Kannada fine-tuned) | data2vec-AQC (multilingual SSL) | XEUS | ESPnet Transformer | Whisper large-v3 |
|---|---|---|---|---|---|
| FLEURS | **21.40** | 23.97 | 25.02 | 43.22 | 25.31 |
| IndicTTS | **18.06** | 22.70 | 23.63 | 28.69 | 23.53 |
| Kathbath | **17.03** | 21.01 | 24.64 | 30.99 | 24.75 |
| Kathbath-Noisy | **17.87** | 22.20 | 29.70 | 44.94 | 30.87 |

**data2vec-AQC (Kannada CTC fine-tuned)** wins on every testset — expected,
since it's the only encoder among the five that was already fine-tuned on
Kannada speech (via CTC) before this SLAM-LLM stage; the rest are
general-purpose SSL/supervised encoders seeing Kannada for the first time
here. Whisper large-v3 (multilingual supervised, 635M-param encoder) is the
best of the remaining four, closely matching XEUS. The ESPnet Transformer
(45M params, far smaller and less pretrained than the others) trails
noticeably, with high insertion counts suggesting less stable generation.

## Training accuracy trend

All five encoders were evaluated during training on the held-out dev set
(`eval_acc`, teacher-forced next-token accuracy — not the same metric as WER
above, but a useful convergence signal):

| Encoder | Final eval_acc (epoch 1) | Notes |
|---|---|---|
| data2vec-AQC (Kannada fine-tuned) | 88.0% | Sharp "emergence" jump ~step 19-20k (59%→88%) |
| data2vec-AQC (multilingual SSL) | 87.25% | Sharp "emergence" jump ~step 19-20k (59%→87%) |
| XEUS | 87.0% (epoch 1: 86.75%, epoch 2 plateau ~87%) | Emergence jump ~step 11k (55.6%→79.6%), earlier than the other two, after fixing `dataset_config.normalize` and switching to the official `SSLTask.build_model_from_file()` loader (100% of checkpoint params load, vs. ~98.3% with an earlier manual-patch workaround) |
| ESPnet Transformer (45M) | 84.9% (step 18k) | Emergence jump ~step 15k (57.7%→78.4%). Critical fix: this encoder's native frame rate is ~31.25fps (512 samples/frame), not the dataset's default 50fps assumption -- `dataset_config.audio_frames_per_sec=31.25` corrects the reserved audio-token-slot budget (without it, ~1.65x too many slots were reserved per utterance) |
| Whisper large-v3 | 86.8% (epoch 1) | Emergence jump ~step 17-18k (56.3%→79.1%). Critical fix: openai-whisper's own `LayerNorm` subclass casts its *input* to float32 but not its own `weight`/`bias`, which DeepSpeed's bf16 mode casts to bf16 for every parameter (including this frozen encoder) -- `"expected scalar type Float but found BFloat16"` crash at the very first training step. Fixed in `slam_model.py`'s existing LayerNorm-patch mechanism by also detecting and patching `whisper.model.LayerNorm` instances (previously only stock `nn.LayerNorm` was patched; whisper's subclass overrides `forward()` so it fell through the existing check) |

## Reproducing

```bash
# Decode (per encoder, 4 testsets each)
bash examples/asr_librispeech/scripts/inference_data2vec_gemma3_kannada_4testsets.sh             # data2vec-AQC (Kannada FT)
bash examples/asr_librispeech/scripts/inference_data2vec_gemma3_kannada_ssl_4testsets.sh         # data2vec-AQC (SSL)
bash examples/asr_librispeech/scripts/inference_data2vec_gemma3_kannada_xeus_4testsets.sh        # XEUS
bash examples/asr_librispeech/scripts/inference_data2vec_gemma3_kannada_transformer_4testsets.sh # ESPnet Transformer
bash examples/asr_librispeech/scripts/inference_data2vec_gemma3_kannada_whisper_4testsets.sh     # Whisper large-v3

# Score
bash examples/asr_librispeech/scripts/compute_wer_kannada_3encoders.sh
```
