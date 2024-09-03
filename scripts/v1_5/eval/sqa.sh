#!/bin/bash

# python -m llava.eval.model_vqa_science \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder ./playground/data/eval/scienceqa/images/test \
#     --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_science_qa.py \
#     --base-dir ./playground/data/eval/scienceqa \
#     --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
#     --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
#     --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json

CKPT=$1
mp=$2
path_to_all_results=$3
MODEL_BASE=$4

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"

read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./llava/eval/model_vqa_science.py \
        --model-path $mp \
        --model-base $MODEL_BASE \
        --question-file ./benchmarks/ScienceQA/llava_test_CQM-A.json \
        --image-folder ./benchmarks/ScienceQA/images/test \
        --answers-file ./playground/data/eval/scienceqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/scienceqa/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/scienceqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python ./benchmarks/ScienceQA/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file $output_file \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}_result.json \
    --path_to_all_results $path_to_all_results
