#!/bin/bash

# bash /mntcephfs/lab_data/songdingjie/mllm/LLaVA/eval_all_benchmarks2.sh
module load cuda11.8/toolkit/11.8.0
# source /mntcephfs/data/med/songdingjie/envs/llava/
cd /mntcephfs/lab_data/songdingjie/mllm/LLaVA/
# export PYTHONPATH="${PYTHONPATH}:/mntcephfs/lab_data/songdingjie/mllm/LLaVA"
# export PATH="${PATH}:/mntcephfs/lab_data/songdingjie/mllm/LLaVA"
# export CUDA_VISIBLE_DEVICES=0

########################## Setup model_id and checkpoint_dir ##########################
declare -A model_dict


## Inference
# model_dict["llava"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5"
# model_dict["llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K448"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K448"
# model_dict["llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K384"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K384"
# model_dict["llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K256"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K256"
# model_dict["llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K144"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K144"
# model_dict["llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K128"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K128"
# model_dict["llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K64"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K64"
# model_dict["llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K32"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K32"
# model_dict["llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K16"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K16"
# model_dict["llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K8"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K8"
# model_dict["llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K4"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K4"
# model_dict["llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K2"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-Pooling_beforeMLP_K2"
# model_dict["llava-vicuna-7b-v1.5-only-inference-PCA_beforeMLP_T0.9"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-PCA_beforeMLP_T0.9"
# model_dict["llava-vicuna-7b-v1.5-only-inference-PCA_beforeMLP_T0.99"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-PCA_beforeMLP_T0.99"
# model_dict["llava-vicuna-7b-v1.5-only-inference-PCA_beforeMLP_T0.999"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-PCA_beforeMLP_T0.999"
# model_dict["llava-vicuna-7b-v1.5-only-inference-PCA_beforeMLP_T0.9999"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-PCA_beforeMLP_T0.9999"
# TextSim
# model_dict["llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.01"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.01"
# model_dict["llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.05"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.05"
# model_dict["llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.1"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.1"
# model_dict["llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.3"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.3"
# model_dict["llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.5"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.5"
# model_dict["llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.7"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.7"
# model_dict["llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.9"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T0.9"
# model_dict["llava-vicuna-7b-v1.5-only-inference-TextSim+_beforeMLP_T0.05"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-TextSim+_beforeMLP_T0.05"
# model_dict["llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T-1"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-TextSim_beforeMLP_T-1"
# model_dict["llava-vicuna-7b-v1.5-only-inference-TextSim+_beforeMLP_T-1"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/inference/llava-vicuna-7b-v1.5-only-inference-TextSim+_beforeMLP_T-1"

## Finetuned
# # Pooling
# model_dict["llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K448"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K448"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K384"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K384"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K256"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K256"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K144"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K144"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K128"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K128"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K64"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K64"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K32"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K32"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K16"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K16"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K8"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K8"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K4"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K4"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K2"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-Pooling_beforeMLP_K2"
# # PCA
# model_dict["llava-vicuna-7b-v1.5-only-finetune-PCA_beforeMLP_T0.9"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-PCA_beforeMLP_T0.9"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-PCA_beforeMLP_T0.99"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-PCA_beforeMLP_T0.99"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-PCA_beforeMLP_T0.999"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-PCA_beforeMLP_T0.999"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-PCA_beforeMLP_T0.9999"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-PCA_beforeMLP_T0.9999"
# # Cluster
# model_dict["llava-vicuna-7b-v1.5-only-finetune-Cluster_beforeMLP_K0.3"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-Cluster_beforeMLP_K0.3"
# TextSim
# model_dict["llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.01"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.01"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.05"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned2/llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.05"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.1"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.1"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.3"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.3"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.5"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.5"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.7"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.7"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.9"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned2/llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T0.9"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-TextSim+_beforeMLP_T0.05"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned2/llava-vicuna-7b-v1.5-only-finetune-TextSim+_beforeMLP_T0.05"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T-1"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned2/llava-vicuna-7b-v1.5-only-finetune-TextSim_beforeMLP_T-1"
# model_dict["llava-vicuna-7b-v1.5-only-finetune-TextSim+_beforeMLP_T-1"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned2/llava-vicuna-7b-v1.5-only-finetune-TextSim+_beforeMLP_T-1"
# model_dict["llava-vicuna-13b-v1.5-only-finetune-TextSim_beforeMLP_T-1"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-13b-v1.5-only-finetune-TextSim_beforeMLP_T-1"
# model_dict["llava-vicuna-13b-v1.5-only-finetune-TextSim_beforeMLP_T0.01"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-13b-v1.5-only-finetune-TextSim_beforeMLP_T0.01"
# model_dict["llava-vicuna-13b-v1.5-only-finetune-TextSim_beforeMLP_T0.05"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-13b-v1.5-only-finetune-TextSim_beforeMLP_T0.05"
# model_dict["llava-vicuna-13b-v1.5-only-finetune-TextSim_beforeMLP_T0.3"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-13b-v1.5-only-finetune-TextSim_beforeMLP_T0.3"
# model_dict["llava-vicuna-13b-v1.5-only-finetune-TextSim+_beforeMLP_T-1"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-13b-v1.5-only-finetune-TextSim+_beforeMLP_T-1"
# model_dict["llava-vicuna-13b-v1.5-only-finetune-TextSim+_beforeMLP_T-1_Lora"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-13b-v1.5-only-finetune-TextSim+_beforeMLP_T-1_Lora"
model_dict["llava-vicuna-13b-v1.5-only-finetune-TextSim+_beforeMLP_T-1_Lora_2e-5"]="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/checkpoints/finetuned/llava-vicuna-13b-v1.5-only-finetune-TextSim+_beforeMLP_T-1_Lora_2e-5"


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

    PATH_TO_ALL_RESULTS="/mntcephfs/lab_data/songdingjie/mllm/LLaVA/benchmark_results/$MODEL_ID.txt"

    ########################## Run each benchmark sequentially ##########################
    # GQA Done
    echo "=========================================================="
    echo "$MODEL_ID Running GQA"
    echo "=========================================================="
    bash scripts/v1_5/eval/gqa.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # MMBench-en
    echo "=========================================================="
    echo "$MODEL_ID Running MMBench-en"
    echo "=========================================================="
    bash scripts/v1_5/eval/mmbench.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # MMBench-cn
    echo "=========================================================="
    echo "$MODEL_ID Running MMBench-cn"
    echo "=========================================================="
    bash scripts/v1_5/eval/mmbench_cn.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # MME
    echo "=========================================================="
    echo "$MODEL_ID Running MME"
    echo "=========================================================="
    bash scripts/v1_5/eval/mme.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # ScienceQA
    echo "=========================================================="
    echo "$MODEL_ID Running ScienceQA"
    echo "=========================================================="
    bash scripts/v1_5/eval/sqa.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # TextVQA
    echo "=========================================================="
    echo "$MODEL_ID Running TextVQA"
    echo "=========================================================="
    bash scripts/v1_5/eval/textvqa.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # SEEDBench
    echo "=========================================================="
    echo "$MODEL_ID Running SEEDBench"
    echo "=========================================================="
    bash scripts/v1_5/eval/seed.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # MM-Vet
    # gpt-4-0613
    echo "=========================================================="
    echo "$MODEL_ID Running MM-Vet"
    echo "=========================================================="
    # bash benchmarks/MM-Vet/eval_mmvet.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE
    bash scripts/v1_5/eval/mmvet.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # POPE
    echo "=========================================================="
    echo "$MODEL_ID Running POPE"
    echo "=========================================================="
    bash scripts/v1_5/eval/pope.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # VQAV2
    echo "=========================================================="
    echo "$MODEL_ID Running VQAV2"
    echo "=========================================================="
    bash scripts/v1_5/eval/vqav2.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # VisWiz
    echo "=========================================================="
    echo "$MODEL_ID Running VisWiz"
    echo "=========================================================="
    bash scripts/v1_5/eval/vizwiz.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # LLaVA-Bench
    echo "=========================================================="
    echo "$MODEL_ID Running LLaVA-Bench"
    echo "=========================================================="
    bash scripts/v1_5/eval/llavabench.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # MSVD
    echo "=========================================================="
    echo "$MODEL_ID Running MSVD"
    echo "=========================================================="
    bash scripts/v1_5/eval/msvd.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # MSRVTT
    echo "=========================================================="
    echo "$MODEL_ID Running MSRVTT"
    echo "=========================================================="
    bash scripts/v1_5/eval/msrvtt.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE

    # # ActivityNet
    # echo "=========================================================="
    # echo "$MODEL_ID Running ActivityNet"
    # echo "=========================================================="
    # bash scripts/v1_5/eval/activitynet.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE






    # # MMMU
    # echo "=========================================================="
    # echo "$MODEL_ID Running MMMU"
    # echo "=========================================================="
    # bash benchmarks/MMMU/eval_mmmu.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE


    # # TouchStone
    # # gpt-4-0613
    # echo "=========================================================="
    # echo "$MODEL_ID Running TouchStone"
    # echo "=========================================================="
    # bash benchmarks/touchstone/eval_touchstone.sh $MODEL_ID $MODLE_DIR $PATH_TO_ALL_RESULTS $MODEL_BASE
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
