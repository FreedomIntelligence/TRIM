#!/bin/bash

##################### WANDB Log ####################
export WANDB_API_KEY=""
export WANDB_PROJECT=""

##################### Data ####################
data_qa_path=""
data_qa_image_folder_path=""

##################### Model ###################
MODEL_VERSION="vicuna-7b-v1.5"
vision_tower="clip-vit-large-patch14-336"
################## Training Config ################
per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=4
model_max_length=2048
########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=v1
########### DO NOT CHANGE ###########
reduce_func=TRIM
mlp_adapter_path="/path/to/llava-vicuna-7b-v1.5-pretrain/"

for reduce_func_param in -1; do
experiment_name=llava-${MODEL_VERSION}-${reduce_func}_T${reduce_func_param}
export WANDB_NAME="${reduce_func}_beforeMLP_T${reduce_func_param}"

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./model/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ${data_qa_path} \
    --image_folder ${data_qa_image_folder_path} \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --vision_tower ${vision_tower} \
    --pretrain_mm_mlp_adapter ${mlp_adapter_path}/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_vision_token_reduce_func ${reduce_func}:${reduce_func_param} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_projector_type mlp2x_gelu \
    --bf16 True \
    --output_dir ./checkpoints/finetuned/${experiment_name} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    > logs/${experiment_name}.log
done
