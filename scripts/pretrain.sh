#!/bin/bash
# Ref: /mntcephfs/lab_data/chenshunian/workspace/LLaVA/scripts/pretrain.sh
# cd /mntcephfs/lab_data/songdingjie/mllm/LLaVA/
# conda activate llava
module load cuda11.8/toolkit/11.8.0
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

##################### WANDB Log ####################
export WANDB_API_KEY="45718b8ebe74972592bd007698ab65de84cf558c"
export WANDB_PROJECT="MLLM"
# export WANDB_NAME="test"
# export WANDB_NOTES="Test training script for LLaVA."
# WANDB_MODE="offline"

##################### Data ####################
data_caption_path="/mntnfs/med_data5/guimingchen/datasets/llava/pretrain/blip_laion_cc_sbu_558k.json"
data_caption_image_folder_path="/mntnfs/med_data5/guimingchen/datasets/llava/pretrain/images"
# data_qa_path="/mntcephfs/data/med/guimingchen/workspaces/vllm/data/laion_caption_qa_gpt4v-web/processed/laion4v_qa_513k.json"
# /mntcephfs/data/med/zhanghongbo/yaojishi/huatuov/datasets/vflan4v_cap_202k.json
# /mntcephfs/data/med/zhanghongbo/yaojishi/huatuov/datasets/vflan4v_cap_202k_images

##################### Model ###################
# MODEL_VERSION=vicuna-v1-3-7b
# MODEL_VERSION=llama-2-7b-chat
# MODEL_VERSION=Meta-Llama-3-8B
# MODEL_VERSION=vicuna-7b-v1.1
MODEL_VERSION=vicuna-7b-v1.5
# vision_tower="model/clip_vit_large_patch14"
vision_tower="model/clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"

################## Training Config ################
per_device_train_batch_size=8
per_device_eval_batch_size=4
gradient_accumulation_steps=8
model_max_length=2048
# model_max_length=4096
########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

experiment_name=llava-${MODEL_VERSION}-pretrain-PCA_beforeMLP_thres0.995

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./model/${MODEL_VERSION} \
    --version $PROMPT_VERSION \
    --data_path ${data_caption_path} \
    --image_folder ${data_caption_image_folder_path} \
    --vision_tower ${vision_tower} \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type mlp2x_gelu \
    --bf16 True \
    --output_dir ./checkpoints/${experiment_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    > logs/${experiment_name}.log
