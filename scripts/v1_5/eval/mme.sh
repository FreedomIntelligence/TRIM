#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/MME/llava_mme.jsonl \
#     --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
#     --answers-file ./playground/data/eval/MME/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# cd ./playground/data/eval/MME

# python convert_answer_to_mme.py --experiment llava-v1.5-13b

# cd eval_tool

# python calculation.py --results_dir answers/llava-v1.5-13b

CKPT=$1
mp=$2
path_to_all_results=$3
MODEL_BASE=$4

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"

read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

CHUNKS=${#GPULIST[@]}


> ./playground/data/eval/MME/answers/$CKPT.jsonl

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}  python ./playground/data/eval/MME/model_vqa_loader.py \
        --model-path $mp \
        --model-base $MODEL_BASE \
        --question-file ./benchmarks/MME/llava_mme.jsonl \
        --image-folder ./benchmarks/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/$CKPT.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &

done

wait

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT --path_to_all_results $path_to_all_results
