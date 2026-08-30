# SLAM-LLM for Kannada-English Bilingual ASR

Bilingual Kannada-English Automatic Speech Recognition (ASR) built on the [SLAM-LLM](https://github.com/ddlBoJack/SLAM-LLM) framework, for Prof. Umesh S's speech-encoder comparison study at IIT Madras SPRING Lab.

A frozen speech encoder feeds a linear projector into **Gemma-3-4B-IT** (fine-tuned with LoRA), trained on Kannada, English, and Kannada-English code-switched data.

## Architecture

```
Raw Audio (16kHz)
    │
    ▼
┌─────────────────────────────────────┐
│   Speech Encoder                ❄️   │  ← FROZEN (one of 5, see below)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│   Projector: Linear             🔥   │  ← TRAINABLE (5x downsample rate)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│   LLM Decoder: Gemma-3-4B-IT    🔥   │  ← TRAINABLE (LoRA)
│   LoRA rank     : 8                 │
│   LoRA alpha    : 32                │
│   LoRA targets  : q/k/v/o/gate/     │
│                   up/down_proj      │
└─────────────────────────────────────┘
    │
    ▼
  Text Transcription
```

Only the text decoder of the multimodal `google/gemma-3-4b-it` checkpoint is loaded (the SigLIP vision tower is dropped), so LoRA attaches only to the text-side projections.

## Encoders compared

| # | Encoder | Type | Notes |
|---|---|---|---|
| 1 | data2vec-AQC | Kannada CTC fine-tuned | SPRING-INX checkpoint |
| 2 | data2vec-AQC | Multilingual SSL pretrained-only | SPRING-INX checkpoint |
| 3 | Custom Transformer | — | In progress |
| 4 | [XEUS](https://huggingface.co/espnet/xeus) | ESPnet2 E-Branchformer SSL, 577M | Official `SSLTask.build_model_from_file()` loader; no input pre-normalization (matches the model's own frontend) |
| 5 | [Wav2Vec2-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0) | Meta Conformer SSL, 580M | `encoder_frame_offset` fix for correct audio-token slot alignment |

All encoders are frozen; only the projector and LoRA adapters are trained.

## Data

Kannada + English + Kannada-English code-switched training data, proportionately merged (~4:1 Kannada:English by hours), with a `kn-en` code-switch prompt variant that tells the LLM when an utterance mixes both languages.

## Notable fixes made during encoder integration

- **XEUS**: our `espnet` package (202402) predated the checkpoint's schema, requiring a manual config-patching workaround that left ~1.7% of encoder params randomly initialized. Upgrading to `espnet==202506` let the official loader load 100% of params directly. Also fixed `dataset_config.normalize` being incorrectly set to `true` — XEUS's own frontend expects raw, unnormalized waveform.
- **Wav2Vec2-BERT 2.0**: the dataset's audio-token-slot reservation formula (`samples // 320`) assumed a raw-CNN encoder frame rate; Wav2Vec2-BERT's log-mel + stride-2 feature-stacking pipeline is consistently 1 frame short of that formula, verified empirically across multiple audio durations. Without the fix, every training example had one reserved slot scattered as an all-zero embedding into the LLM's input. Fixed via `dataset_config.encoder_frame_offset`.

## Training

- Framework: DeepSpeed ZeRO-2, 4 GPUs
- Effective batch size: 4 (micro) × 2 (grad accum) × 4 (GPUs) = 32
- Checkpoint/eval cadence decoupled (`checkpoint_interval`, `max_eval_steps`) so validation can run frequently without paying the full-dev-set cost every time

See `examples/asr_librispeech/scripts/finetune_data2vec_gemma3_kannada_*.sh` for the per-encoder training scripts and `examples/asr_librispeech/scripts/inference_data2vec_gemma3_kannada.sh` for benchmark decoding (FLEURS, IndicTTS, Kathbath, Kathbath-Noisy).

## Acknowledgements

Built on [SLAM-LLM](https://github.com/ddlBoJack/SLAM-LLM). Encoder checkpoints from [SPRING Lab](https://asr.iitm.ac.in/), [ESPnet](https://github.com/espnet/espnet), and [Meta AI](https://huggingface.co/facebook/w2v-bert-2.0).
