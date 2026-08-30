#!/bin/bash
# Train: Wav2Vec2-BERT 2.0 (Meta, 580M, 4.5M-hour pretrained-only, frozen) +
# linear projector + gemma-3-4b-it (LoRA)
# Bilingual Kannada-English code-switch experiment — encoder #5 of 5 in
# Prof. Umesh's encoder comparison study. Identical downstream setup to
# finetune_data2vec_gemma3_kannada_cs.sh (encoder #1) except for the encoder
# (name=wav2vec2_bert, dim=1024) + output_dir + wandb exp name.
#
# See src/slam_llm/models/wav2vec2_bert_encoder.py — unlike the raw-waveform
# encoders, this one does its own log-mel feature extraction internally
# (SeamlessM4TFeatureExtractor) before calling the HF Wav2Vec2BertModel.

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

export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

# Re-check nvidia-smi before launching — pick 4 free GPUs (encoders #1/#2/#4
# currently use 1,2,3,4 and 6,7,8,9; adjust below to whatever is actually free).
export CUDA_VISIBLE_DEVICES=${GPU_LIST:-1,2,3,4}

run_dir=/speech/abhishek/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/speech/abhishek/kannada_data/encoders/wav2vec2-bert-2.0
llm_path=/speech/abhishek/SLAM_Hindi/models/google/gemma-3-4b-it
ds_config_path=$run_dir/examples/asr_librispeech/conf/ds_config_gemma3_bilingual.json

train_data_path=/speech/abhishek/kannada_data/merged/train_kn_en_merged.jsonl
val_data_path=/speech/abhishek/kannada_data/merged/dev_kn_en_merged.jsonl

output_dir=${OUTPUT_DIR:-/speech/abhishek/output/kannada-cs-w2vbert-normed-gemma3-4b-finetuned}
mkdir -p "$output_dir"

n_gpus=4

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
    echo "[resume] no complete checkpoint found — starting fresh run in $output_dir"
fi

lora_targets=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]

resume_arg=""
if [ -n "$resume_ckpt" ]; then
    resume_arg="++train_config.resume_ckpt=$resume_ckpt"
fi

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=gemma-3-4b-it \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=2560 \
++model_config.encoder_name=wav2vec2_bert \
++model_config.normalize=false \
++dataset_config.normalize=false \
++dataset_config.encoder_frame_offset=1 \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1024 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=raw \
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
++log_config.wandb_exp_name=wav2vec2-bert-2.0-normed-pretrained-kannada-bilingual-cs-gemma3-4b-it-lora \
++log_config.log_interval=5 \
"

/speech/abhishek/miniconda3/envs/slam_llm/bin/deepspeed \
    --include=localhost:${GPU_LIST:-1,2,3,4} \
    --master_port=29511 \
    $code_dir/deepspeed_finetune_asr.py \
    --config-path "conf" \
    --config-name "prompt_kannada_ctx.yaml" \
    $hydra_args
