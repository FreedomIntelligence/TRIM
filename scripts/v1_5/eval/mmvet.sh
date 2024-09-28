#!/bin/bash

CKPT=$1
mp=$2
path_to_all_results=$3
MODEL_BASE=$4

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"

read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

CHUNKS=${#GPULIST[@]}

################### chunked ###################
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./llava/eval/model_vqa.py \
        --model-path $mp \
        --model-base $MODEL_BASE \
        --question-file ./benchmarks/MM-Vet/llava-mm-vet.jsonl \
        --image-folder ./benchmarks/MM-Vet/mm-vet/images \
        --answers-file ./playground/data/eval/mm-vet/answers/${CKPT}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait
################### chunked ###################

output_file=./playground/data/eval/mm-vet/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mm-vet/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


mkdir -p ./playground/data/eval/mm-vet/results

python ./benchmarks/MM-Vet/convert_mmvet_for_eval.py \
    --src $output_file \
    --dst ./playground/data/eval/mm-vet/results/$CKPT.json

cd ./playground/data/eval/mm-vet
python mm-vet_evaluator.py "$CKPT" "$path_to_all_results"
