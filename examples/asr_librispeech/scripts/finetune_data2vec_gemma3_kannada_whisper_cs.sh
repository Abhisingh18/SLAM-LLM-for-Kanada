#!/bin/bash
# Train: Whisper large-v3 encoder (OpenAI, 1280-dim, frozen) + linear
# projector + gemma-3-4b-it (LoRA)
# Bilingual Kannada-English code-switch experiment -- encoder #6 in
# Prof. Umesh's encoder comparison study. Uses SLAM-LLM's NATIVE Whisper
# support (WhisperWrappedEncoder in src/slam_llm/models/encoder.py +
# slam_model.py's "whisper" dispatch) -- this is the base framework's own
# feature, unmodified; confirmed identical to Akshaya's reference setup
# (same encoder.py/slam_model.py code, she didn't patch either). We use our
# OWN speech_dataset.py (not her custom speech_dataset_lang.py) -- it already
# supports input_type=mel (Whisper log-mel extraction) + use_history_context,
# same pattern as the other 5 Kannada encoders in this study.
#
# input_type=mel (NOT raw) -- Whisper's own log-mel frontend is used via
# speech_dataset.py's _whisper_mod.log_mel_spectrogram() call, so audio is
# pre-converted to mel here rather than inside the encoder. mel_size=128
# matches large-v3 (the v1/v2 checkpoints use 80 mels; explicitly set to
# avoid the dataset's default of 80 silently mismatching the encoder).

export PYTHONPATH=/speech/abhishek/fairseq:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:/speech/abhishek/miniconda3/envs/slam_llm/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DS_SKIP_CUDA_CHECK=1
export HYDRA_FULL_ERROR=1

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# Stability mitigations for recurring GPU-level faults (illegal instruction /
# CUBLAS_STATUS_EXECUTION_FAILED): disable TF32 tensor-core paths and force
# deterministic cuDNN algorithm selection. Slightly slower, more stable.
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

# Re-check nvidia-smi before launching -- pick 4 free GPUs.
export CUDA_VISIBLE_DEVICES=${GPU_LIST:-6,7,8,9}

run_dir=/speech/abhishek/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/speech/abhishek/kannada_data/encoders/whisper/large-v3.pt
llm_path=/speech/abhishek/SLAM_Hindi/models/google/gemma-3-4b-it
ds_config_path=$run_dir/examples/asr_librispeech/conf/ds_config_gemma3_bilingual.json

train_data_path=/speech/abhishek/kannada_data/merged/train_kn_en_merged.jsonl
val_data_path=/speech/abhishek/kannada_data/merged/dev_kn_en_merged.jsonl

# Fixed (not timestamped) output_dir -- re-running this same script always
# targets the same dir, so wandb_run_id.txt is found automatically and the
# run continues on the same wandb graph, and checkpoint auto-resume below
# can find prior checkpoints.
output_dir=${OUTPUT_DIR:-/speech/abhishek/output/kannada-cs-whisper-gemma3-4b-finetuned}
mkdir -p "$output_dir"

n_gpus=4  # must match the number of GPUs in CUDA_VISIBLE_DEVICES / --include below

# Auto-detect the latest COMPLETE checkpoint under $output_dir to resume from.
# A checkpoint is complete only if it has pytorch_model.bin, a `latest` file,
# AND all n_gpus rank optimizer-state shards (partial/corrupt checkpoints from
# a crash mid-save are skipped). Override with RESUME_CKPT=... to force a
# specific checkpoint, or RESUME_CKPT=none to force a fresh run.
resume_ckpt=""
if [ "${RESUME_CKPT:-}" = "none" ]; then
    resume_ckpt=""
elif [ -n "${RESUME_CKPT:-}" ]; then
    resume_ckpt="$RESUME_CKPT"
else
    for d in $(ls -dt "$output_dir"/asr_epoch_*_step_* 2>/dev/null); do
        if [ -f "$d/pytorch_model.bin" ] && [ -f "$d/latest" ]; then
            gstep_dir="$d/$(cat "$d/latest" 2>/dev/null)"
            n_optim=$(ls "$gstep_dir"/bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt 2>/dev/null | wc -l)
            if [ "$n_optim" -eq "$n_gpus" ]; then
                resume_ckpt="$d"
                break
            fi
        fi
    done
fi

if [ -n "$resume_ckpt" ]; then
    echo "[resume] found complete checkpoint, resuming from: $resume_ckpt"
else
    echo "[resume] no complete checkpoint found -- starting fresh run in $output_dir"
fi

lora_targets=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]

resume_arg=""
if [ -n "$resume_ckpt" ]; then
    # Weight restore (ckpt_path, top-level kwarg -> slam_model.py's
    # setup_model()) + step/epoch counter & wandb-graph continuity
    # (train_config.resume_ckpt, parsed by finetune_deepspeed.py to
    # fast-forward the dataloader) -- see the fixes made for the W2VBert
    # resume bug earlier in this project.
    resume_arg="++ckpt_path=$resume_ckpt/pytorch_model.bin ++train_config.resume_ckpt=$resume_ckpt"
fi

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=gemma-3-4b-it \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=2560 \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=128 \
++dataset_config.normalize=true \
++dataset_config.prompt_style=gemma2 \
++dataset_config.use_history_context=true \
++train_config.model_name=asr \
++train_config.num_epochs=10 \
++train_config.enable_deepspeed=true \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.use_peft=true \
++train_config.peft_config.peft_method=lora \
++train_config.peft_config.r=8 \
++train_config.peft_config.lora_alpha=32 \
++train_config.peft_config.target_modules=$lora_targets \
++train_config.peft_config.lora_dropout=0.05 \
++train_config.peft_config.bias=none \
++train_config.peft_config.task_type=CAUSAL_LM \
++train_config.batching_strategy=custom \
++train_config.use_fp16=false \
++train_config.warmup_steps=1000 \
++train_config.total_steps=200000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.checkpoint_interval=1000 \
++train_config.max_eval_steps=500 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=8 \
++train_config.output_dir=$output_dir \
$resume_arg \
++deepspeed_config=$ds_config_path \
++metric=acc \
++log_config.log_file=$output_dir/train.log \
++log_config.use_wandb=true \
++log_config.wandb_dir=/speech/abhishek/logs/wandb \
++log_config.wandb_entity_name=abhisingh964800-iit-madras-foundation \
++log_config.wandb_project_name=Kannada_English_Encoder_Comparison \
++log_config.wandb_exp_name=whisper-large-v3-kannada-bilingual-cs-gemma3-4b-it-lora \
++log_config.log_interval=5 \
"

/speech/abhishek/miniconda3/envs/slam_llm/bin/deepspeed \
    --include=localhost:${GPU_LIST:-6,7,8,9} \
    --master_port=29513 \
    $code_dir/deepspeed_finetune_asr.py \
    --config-path "conf" \
    --config-name "prompt_kannada_ctx.yaml" \
    $hydra_args
