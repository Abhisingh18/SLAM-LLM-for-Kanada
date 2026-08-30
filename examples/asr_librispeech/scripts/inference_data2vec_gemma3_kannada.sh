#!/bin/bash
# Inference: Data2Vec-AQC (Kannada fine-tuned) + Gemma-3-4B-IT + LoRA
# Decodes the held-out Kannada+English dev set (no separate benchmark
# testsets exist yet for Kannada — ask Akshaya, then extend the testsets
# array below the same way the Hindi-English 9-testset script does).
#
# Usage: CKPT_DIR=<checkpoint dir> bash inference_data2vec_gemma3_kannada.sh
# Defaults to the best/latest encoder-#1 checkpoint (Kannada fine-tuned,
# step_33000, ~88% eval_acc) if CKPT_DIR is not set.

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=/speech/abhishek/fairseq:$PYTHONPATH
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:/speech/abhishek/miniconda3/envs/slam_llm/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HYDRA_FULL_ERROR=1

run_dir=/speech/abhishek/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/speech/abhishek/kannada_data/encoders/SPRING_INX_data2vec_aqc_Kannada.pt
llm_path=/speech/abhishek/SLAM_Hindi/models/google/gemma-3-4b-it

output_dir=/speech/abhishek/output/kannada-cs-data2vec-gemma3-4b-finetuned
ckpt_dir=${CKPT_DIR:-$output_dir/asr_epoch_1_step_33000}

if [ ! -f "$ckpt_dir/pytorch_model.bin" ]; then
    echo "ckpt not found: $ckpt_dir/pytorch_model.bin" >&2
    exit 1
fi

decode_dir=$output_dir/decode_results_$(basename $ckpt_dir)
mkdir -p $decode_dir

test_dir=/speech/abhishek/kannada_data/testsets/jsonl
testsets=(
    fleurs_kannada_test
    indictts_kannada_test
    kathbath_kannada_test
    kathbath_noisy_kannada_test
)

lora_targets=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]

for split in "${testsets[@]}"; do
    echo "========================================================"
    echo "Decoding: $split"
    echo "========================================================"

    val_data_path=$test_dir/${split}.jsonl
    decode_log=$decode_dir/decode_${split}

    /speech/abhishek/miniconda3/envs/slam_llm/bin/python $code_dir/inference_asr_batch.py \
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

    echo "Done: $split"
    echo "GT  : ${decode_log}_gt"
    echo "PRED: ${decode_log}_pred"

    python3 $code_dir/compute_wer_hindi_v2.py \
        ${decode_log}_gt \
        ${decode_log}_pred \
        ${decode_log}_wer
    echo "=== WER ($split) ==="
    grep -E "^%WER|^%SER|^Scored" ${decode_log}_wer
    echo ""
done

echo "All testsets done! Results in: $decode_dir/"
