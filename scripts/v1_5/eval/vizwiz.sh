#!/bin/bash
CUDA_VISIBLE_DEVICES=0
CKPT=$1
mp=$2
path_to_all_results=$3
MODEL_BASE=$4

python ./llava/eval/model_vqa_loader.py \
    --model-path $mp \
    --model-base $MODEL_BASE \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$CKPT.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$CKPT.json



