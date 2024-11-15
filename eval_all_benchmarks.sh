#!/bin/bash

########################## Setup model_id and checkpoint_dir ##########################
declare -A model_dict

## No Finetuned
# model_dict["llava-vicuna-7b-v1.5-only-inference-TRIMllava-v1.5-7b-TRIM "]="/path/to/llava-vicuna-7b-v1.5-only-inference-TRIM"
# model_dict["llava-vicuna-13b-v1.5-only-inference-TRIM"]="/path/to/llava-vicuna-13b-v1.5-only-inference-TRIM"

## Finetuned
model_dict["llava-v1.5-7b-TRIM"]="/path/to/llava-v1.5-7b-TRIM"
# model_dict["llava-v1.5-13b-TRIM"]="/path/to/llava-v1.5-13b-TRIM"

for key in "${!model_dict[@]}"; do
    MODEL_ID=$key
    MODLE_DIR=${model_dict[$key]}
    MODEL_BASE=""

    if [[ "$MODEL_ID" == *"Lora"* ]]; then
        if [[ "$MODEL_ID" == *"13b"* ]]; then
            MODEL_BASE="model/llava-v1.5-13b"
        else
            MODEL_BASE="model/llava-v1.5-7b"
        fi
    fi

    PATH_TO_ALL_RESULTS="/path/tp/benchmark_results/$MODEL_ID.txt"

    ########################## Run each benchmark sequentially ##########################
    # GQA
    echo "=========================================================="
    echo "$MODEL_ID Running GQA"
    echo "=========================================================="
    bash scripts/v1_5/eval/gqa.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # # MMBench-en
    # echo "=========================================================="
    # echo "$MODEL_ID Running MMBench-en"
    # echo "=========================================================="
    # bash scripts/v1_5/eval/mmbench.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # # MMBench-cn
    # echo "=========================================================="
    # echo "$MODEL_ID Running MMBench-cn"
    # echo "=========================================================="
    # bash scripts/v1_5/eval/mmbench_cn.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # # MME
    # echo "=========================================================="
    # echo "$MODEL_ID Running MME"
    # echo "=========================================================="
    # bash scripts/v1_5/eval/mme.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # # ScienceQA
    # echo "=========================================================="
    # echo "$MODEL_ID Running ScienceQA"
    # echo "=========================================================="
    # bash scripts/v1_5/eval/sqa.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # # TextVQA
    # echo "=========================================================="
    # echo "$MODEL_ID Running TextVQA"
    # echo "=========================================================="
    # bash scripts/v1_5/eval/textvqa.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # # SEEDBench
    # echo "=========================================================="
    # echo "$MODEL_ID Running SEEDBench"
    # echo "=========================================================="
    # bash scripts/v1_5/eval/seed.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # # MM-Vet
    # echo "=========================================================="
    # echo "$MODEL_ID Running MM-Vet"
    # echo "=========================================================="
    # bash scripts/v1_5/eval/mmvet.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # # POPE
    # echo "=========================================================="
    # echo "$MODEL_ID Running POPE"
    # echo "=========================================================="
    # bash scripts/v1_5/eval/pope.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # # VQAV2
    # echo "=========================================================="
    # echo "$MODEL_ID Running VQAV2"
    # echo "=========================================================="
    # bash scripts/v1_5/eval/vqav2.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # # VisWiz
    # echo "=========================================================="
    # echo "$MODEL_ID Running VisWiz"
    # echo "=========================================================="
    # bash scripts/v1_5/eval/vizwiz.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # # LLaVA-Bench
    # echo "=========================================================="
    # echo "$MODEL_ID Running LLaVA-Bench"
    # echo "=========================================================="
    # bash scripts/v1_5/eval/llavabench.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # ########################## Run each benchmark sequentially ##########################
done





################################# how to modify codes #################################

# procedure:
<<'###'
0. replace from llava... with from visionjamba

1. in generate.py, import the correct chatbot:
    from visionjamba.eval.chatbot import Chatbot

2. in generate.py, add image placeholder before bot.chat
        qs = "<image>\n" + qs

3. in eval.py,
parser.add_argument('--path_to_all_results', required=True, help="path to all benchmark results, a tsv file")

path_to_all_results=sys.argv[1]

# write the target result to a desired file
if m == 'accuracy':
    with open(args.path_to_all_results, 'a') as f:
        f.write(f"Benchmark_Name\t{score}\n")


4. in bash script, setup environment:

CKPT=$1
mp=$2
path_to_all_results=$3

gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
# gpu_list="2,3,4,5,6"

read -a GPULIST <<< ${gpu_list//,/ }
# GPULIST=(0 1)

CHUNKS=${#GPULIST[@]}


5. in bash script, pass the path to eval.py:
--path_to_all_results $path_to_all_results

6. check the file: /wangbenyou/xidong/VisionJamba/benchmark_results/debug.txt

7. clear the answer folder!
###

################################# how to modify codes #################################
