#!/bin/bash
# Inference: Data2Vec-AQC (Kannada CTC fine-tuned) + Gemma-3-4B-IT + LoRA
# Copied from inference_data2vec_gemma3_bilingual_9testsets.sh (Hindi-English
# setup) and adapted for Kannada: 4 benchmark testsets (FLEURS, IndicTTS,
# Kathbath, Kathbath-Noisy), prompt_kannada_ctx.yaml, and the Kannada
# CTC-finetuned data2vec-aqc checkpoint/encoder path.

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-6}
export PYTHONPATH=/speech/abhishek/fairseq:$PYTHONPATH
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:/speech/abhishek/miniconda3/envs/slam_llm/bin:$PATH
# Stability mitigations for recurring GPU-level faults (illegal instruction /
# CUBLAS_STATUS_EXECUTION_FAILED), same as the training scripts: disable TF32
# tensor-core paths and force deterministic cuDNN algorithm selection.
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HYDRA_FULL_ERROR=1

run_dir=/speech/abhishek/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/speech/abhishek/kannada_data/encoders/SPRING_INX_data2vec_aqc_SSL.pt
llm_path=/speech/abhishek/SLAM_Hindi/models/google/gemma-3-4b-it

output_dir=/speech/abhishek/output/kannada-cs-data2vec-ssl-gemma3-4b-finetuned
ckpt_dir=${CKPT_DIR:-$output_dir/asr_epoch_1_step_33000}

decode_dir=$output_dir/decode_results_$(basename $ckpt_dir)_4testsets
mkdir -p $decode_dir

test_dir=/speech/abhishek/kannada_data/testsets/jsonl

lora_targets=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]

testsets=(
    fleurs_kannada_test
    indictts_kannada_test
    kathbath_kannada_test
    kathbath_noisy_kannada_test
)

for split in "${testsets[@]}"; do
    echo "========================================================"
    echo "Decoding: $split"
    echo "========================================================"

    val_data_path=$test_dir/${split}.jsonl
    decode_log=$decode_dir/decode_${split}

    attempt=1
    max_attempts=5
    until /speech/abhishek/miniconda3/envs/slam_llm/bin/python $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt_kannada_ctx.yaml" \
        hydra.run.dir=$ckpt_dir \
        ++model_config.llm_name=gemma-3-4b-it \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=2560 \
        ++model_config.encoder_name=data2vec_aqc \
        ++model_config.normalize=true \
        ++dataset_config.normalize=true \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1024 \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.prompt_style=gemma2 \
        ++dataset_config.use_history_context=true \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
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
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=4 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_dir/pytorch_model.bin \
        ++log_config.log_file=${decode_log}.log \
        ++log_config.use_wandb=false
    do
        echo "[retry] $split attempt $attempt failed (exit $?)"
        attempt=$((attempt+1))
        if [ $attempt -gt $max_attempts ]; then
            echo "[FATAL] $split failed after $max_attempts attempts, skipping"
            break
        fi
        sleep 10
    done

    echo "Done: $split"
    echo "GT  : ${decode_log}_gt"
    echo "PRED: ${decode_log}_pred"
    echo ""
done

echo "All 4 Kannada testsets done! Results in: $decode_dir/"
