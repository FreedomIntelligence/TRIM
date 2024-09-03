#!/bin/bash

# CUDA_VISIBLE_DEVICES='0,1,2,3'
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

CKPT=$1
mp=$2
path_to_all_results=$3
MODEL_BASE=$4

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"

read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

CHUNKS=${#GPULIST[@]}

OPENAIKEY="sk-Bvt6KOEoa73qQQm6Eb4590C03a2046B4874c0a31F29034B7"
OPENAIBASE="https://api.ai-gaochao.cn/v1"

for IDX in $(seq 0 $((CHUNKS-1))); do
    # MSVD-QA
    CMD="CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./llava/eval/model_vidqa.py \
    --model-path $mp \
    --video_dir /mntcephfs/lab_data/songdingjie/mllm/LLaMA-VID/data/LLaMA-VID-Eval/MSVD-QA/video \
    --gt_file /mntcephfs/lab_data/songdingjie/mllm/LLaMA-VID/data/LLaMA-VID-Eval/MSVD-QA/test_qa.json \
    --output_dir ./playground/data/eval/msvd/answers/${CKPT} \
    --output_name pred \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1"

    if [[ -n $MODEL_BASE ]]; then
        CMD="$CMD --model-base $MODEL_BASE"
    fi

    eval $CMD &
done

wait

python llava/eval/eval_msvd_qa.py \
    --pred_path ./playground/data/eval/msvd/answers/${CKPT} \
    --output_dir ./playground/data/eval/msvd/answers/${CKPT}/results \
    --output_json ./playground/data/eval/msvd/answers/${CKPT}/results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    --api_key $OPENAIKEY \
    --api_base $OPENAIBASE \
    --result_file "$path_to_all_results"
