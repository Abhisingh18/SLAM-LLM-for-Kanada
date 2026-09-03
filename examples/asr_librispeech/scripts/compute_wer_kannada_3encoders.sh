#!/bin/bash
# Compute WER for the Kannada 4-testset decode across all 4 completed
# encoders (data2vec-AQC Kannada-finetuned, data2vec-AQC SSL, XEUS, ESPnet
# Transformer). Copied from compute_wer_bilingual_9testsets.sh; uses
# akshaya_wer.py (Unicode-safe, script-agnostic normalization -- works for
# Kannada too).

py=/speech/abhishek/miniconda3/envs/slam_llm/bin/python3
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script=$script_dir/akshaya_wer.py

testsets=(
    fleurs_kannada_test
    indictts_kannada_test
    kathbath_kannada_test
    kathbath_noisy_kannada_test
)

# encoder_name : decode_dir pairs
declare -A decode_dirs=(
    ["data2vec-FT"]="/speech/abhishek/output/kannada-cs-data2vec-gemma3-4b-finetuned/decode_results_asr_epoch_1_step_33000_4testsets"
    ["data2vec-SSL"]="/speech/abhishek/output/kannada-cs-data2vec-ssl-gemma3-4b-finetuned/decode_results_asr_epoch_1_step_33000_4testsets"
    ["XEUS"]="/speech/abhishek/output/kannada-cs-xeus-v2-tomson-matched-gemma3-4b-finetuned/decode_results_asr_epoch_1_step_33000_4testsets"
    ["Transformer"]="/speech/abhishek/output/kannada-cs-transformer-gemma3-4b-finetuned/decode_results_asr_epoch_1_step_33000_4testsets"
    ["Whisper"]="/speech/abhishek/output/kannada-cs-whisper-gemma3-4b-finetuned/decode_results_asr_epoch_1_step_33000_4testsets"
)

for encoder in "${!decode_dirs[@]}"; do
    decode_dir="${decode_dirs[$encoder]}"
    wer_dir=$decode_dir/wer
    mkdir -p "$wer_dir"

    echo "########################################################"
    echo "Encoder: $encoder"
    echo "########################################################"

    for split in "${testsets[@]}"; do
        gt=$decode_dir/decode_${split}_gt
        pred=$decode_dir/decode_${split}_pred
        out=$wer_dir/wer_${split}

        if [ ! -s "$gt" ] || [ ! -s "$pred" ]; then
            echo "$split: NO RESULT (gt/pred missing or empty)" >&2
            continue
        fi

        $py "$script" "$gt" "$pred" "$out"
    done
done

echo
echo "=== WER summary ($(date +"%H:%M:%S")) ==="
for encoder in "${!decode_dirs[@]}"; do
    decode_dir="${decode_dirs[$encoder]}"
    wer_dir=$decode_dir/wer
    echo "--- $encoder ---"
    for split in "${testsets[@]}"; do
        out=$wer_dir/wer_${split}
        if [ -s "$out" ]; then
            wer=$(grep -m1 "%WER" "$out")
            echo "$split: $wer"
        else
            echo "$split: NO RESULT"
        fi
    done
    echo
done
