#!/bin/bash

# SPLIT="mmbench_dev_cn_20231003"

# python -m llava.eval.model_vqa_mmbench \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
#     --answers-file ./playground/data/eval/mmbench_cn/answers/$SPLIT/llava-v1.5-13b.jsonl \
#     --lang cn \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

# python scripts/convert_mmbench_for_submission.py \
#     --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
#     --result-dir ./playground/data/eval/mmbench_cn/answers/$SPLIT \
#     --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT \
#     --experiment llava-v1.5-13b

CKPT=$1
mp=$2
path_to_all_results=$3
MODEL_BASE=$4

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"

read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

CHUNKS=${#GPULIST[@]}

# mkdir -p ./logs/mmbench/

SPLIT="mmbench_dev_cn_20231003"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./llava/eval/model_vqa_mmbench.py \
        --model-path $mp \
        --model-base $MODEL_BASE \
        --question-file ./benchmarks/MMBench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --single-pred-prompt \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --all-rounds \
        --conv-mode vicuna_v1 & # > ./logs/mmbench/$CKPT.log 2>&1 &
done

wait



output_file=./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python -u ./benchmarks/MMBench/convert_mmbench_for_submission.py \
    --annotation-file ./benchmarks/MMBench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT/ \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT \
    --path_to_all_results $path_to_all_results


# cd /mntcephfs/data/med/guimingchen/workspaces/vllm/LLaVA/benchmarks/MMBench/
# python ./3_out_score_xlsx_cgm.py "$CKPT"
