#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --image-folder ./playground/data/eval/pope/val2014 \
#     --answers-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_pope.py \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl

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
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./playground/data/eval/pope/model_vqa_loader.py \
        --model-path $mp \
        --model-base $MODEL_BASE \
        --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
        --image-folder /mntnfs/med_data5/guimingchen/datasets/MSCOCO/val2014 \
        --answers-file ./playground/data/eval/pope/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX  \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/pope/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/pope/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python ./llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $output_file \
    --path_to_all_results $path_to_all_results
