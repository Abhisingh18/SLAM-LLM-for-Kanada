# Longform ASR for Indian Languages

Automatic Speech Recognition (ASR) for Indian languages (Tamil & Hindi) using the SLAM-LLM framework with Silero VAD-based longform inference and rolling previous context.

---

## Architecture

```
Raw Audio
    │
    ▼
┌─────────────────────────────────────┐
│   Speech Encoder: data2vec-AQC  ❄️   │  ← FROZEN
│   Params : 314.32 M                 │
│   Trainable : 0.00 M               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│   Projector: Linear             🔥   │  ← TRAINABLE
│   Params : 14.68 M                  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│   LLM Decoder: Sarvam-1         🔥   │  ← TRAINABLE (LoRA)
│   Total Params  : 2.52 B            │
│   Trainable     : 11.98 M           │
│   LoRA rank     : 32                │
│   LoRA alpha    : 128               │
│   LoRA targets  : q/k/v/o/gate/     │
│                   up/down proj      │
└─────────────────────────────────────┘
    │
    ▼
  Text Transcription
```

### Parameter Summary

| Component | Total Params | Trainable | Status |
|---|---|---|---|
| data2vec-AQC Encoder | 314.32 M | 0 M | Frozen ❄️ |
| Linear Projector | 14.68 M | 14.68 M | Trainable 🔥 |
| Sarvam-1 LLM (LoRA) | 2,525 M | 11.98 M | Trainable 🔥 |
| **Total** | **~2,854 M** | **~26.66 M** | |

---

## Training

### Setup
- **Data**: Tamil (~2000 hours), Hindi (~4000 hours)
- **Encoder**: `SPRING_INX_data2vec_aqc_Tamil.pt` / `SPRING_INX_data2vec_aqc_Hindi.pt` (frozen)
- **Projector**: Linear layer (5x downsampling rate)
- **LLM**: Sarvam-1 with LoRA fine-tuning
- **Training framework**: DeepSpeed multi-GPU

### LoRA Config
```
r           = 32
lora_alpha  = 128
dropout     = 0.05
bias        = none
target      = q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

### Prompt (with previous context)
```
Using previous context: {prev_context}. Transcribe speech to text. Apply relevant details from the previous context.
```

### Training Scripts
```bash
# Tamil
bash examples/asr_librispeech/scripts/finetune_data2vec_sarvam1_longform.sh

# Hindi
bash examples/asr_librispeech/scripts/finetune_data2vec_sarvam1_longform_hindi.sh
```

---

## Inference (Longform)

### Pipeline

```
Long Audio File
      │
      ▼
 [Step 1] ffmpeg → 16kHz mono WAV
      │
      ▼
 [Step 2] Silero VAD chunking
          threshold     = 0.4
          max chunk     = 15 seconds
          min chunk     = 1 second
      │
      ▼
 [Step 3] Chunk-by-chunk inference
          with 1 previous chunk context
      │
      ▼
 Transcription (tab-separated: key \t text)
```

### Run Inference

```bash
# Tamil
bash examples/asr_librispeech/scripts/run_longform_tamil_1ctx.sh \
    /path/to/audio.wav \
    my_experiment

# Hindi
bash examples/asr_librispeech/scripts/run_longform_hindi_1ctx.sh \
    /path/to/audio.wav \
    my_experiment
```

Output at: `/speech/abhishek/exps/longform/<experiment_name>/decode_1ctx/decode_pred`

---

## Tamil ASR Results (WER %)

### Model Comparison

| Encoder ❄️ | Projector 🔥 | Decoder 🔥 | Steps | CommonVoice | FLEURS | IndictTS | Kathbath | Kathbath Noisy | MUCS | MILE | r1_eval | r2_eval | MIC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| data2vec-AQC (314M) | linear (15.73M) | gemma-3-4b-pt — LoRA (14.90M trainable) | 75K | 7.56 | 12.33 | 4.95 | 5.71 | 6.35 | 7.45 | 4.15 | 10.77 | 11.22 | 6.38 |
| data2vec-AQC (314M) | linear (14.68M) | **Sarvam-1 — LoRA (11.98M trainable)** | 100K | **7.37** | **12.14** | 5.90 | 6.05 | 6.74 | **5.37** | **4.14** | **10.09** | **10.40** | **5.96** |
| data2vec-AQC (314M) | linear (14.68M) | gemma-2b-it-tamil — LoRA (9.80M trainable) | 80K | 7.94 | 11.69 | 5.40 | 5.35 | 6.04 | 6.33 | 4.07 | 11.34 | 12.02 | 5.82 |

### Test Set Details

| Test Set | Sentences |
|---|---|
| CommonVoice Tamil | 5,702 |
| FLEURS Tamil | 591 |
| IndicTTS Tamil | 100 |
| Kathbath Tamil | 1,642 |
| Kathbath Noisy Tamil | 1,642 |
| MUCS Tamil | 2,609 |
| MILE Tamil | 12,087 |
| r1_eval | 3,165 |
| r2_eval | 2,125 |
| MIC | 2,609 |

> **Best overall**: Sarvam-1 wins on CommonVoice, FLEURS, MUCS, MILE, r1_eval, r2_eval, MIC

---

## Hindi ASR Results (WER %)

### Model Comparison

| Encoder ❄️ | Projector 🔥 | Decoder 🔥 | Steps | CommonVoice | FLEURS | IndicTTS | Kathbath | Kathbath Noisy | MUCS |
|---|---|---|---|---|---|---|---|---|---|
| data2vec-AQC (314M) | linear (14.68M) | **Sarvam-1 — LoRA (11.98M trainable)** | epoch1 (~63K) | **6.95** | **7.90** | **5.96** | **5.02** | **5.60** | **7.25** |
| data2vec-AQC (314M) | linear (14.68M) | Airavata — LoRA | 169K | 7.45 | 7.87 | 6.01 | 5.24 | 6.30 | 7.98 |

### Test Set Details

| Test Set | Sentences |
|---|---|
| CommonVoice Hindi | 1,727 |
| FLEURS Hindi | 418 |
| IndicTTS Hindi | 100 |
| Kathbath Hindi | 1,929 |
| Kathbath Noisy Hindi | 1,929 |
| MUCS Hindi | 3,897 |

> **Training Data**: Hindi ~4000 Hours  
> **Best overall**: Sarvam-1 wins on CommonVoice, IndicTTS, Kathbath, Kathbath Noisy, MUCS

---

## Checkpoints

| Language | Checkpoint |
|---|---|
| Tamil | `sarvam-1-tamil-LoRA-prevctx-20260618_151118/asr_epoch_1_step_80000` |
| Hindi | `sarvam-1-hindi-LoRA-prevctx-20260604_193035` |

---

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.10
- PyTorch + CUDA
- Silero VAD (`torch.hub`)
- HuggingFace Transformers
- DeepSpeed (for training)
- ffmpeg
