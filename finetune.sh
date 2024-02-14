#!/bin/bash
            
data_path="finetune.json"
output_path="medalpaca-13b"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=2023 medalpaca/train.py \
    --model "medalpaca/medalpaca-13b" \
    --data_path "$data_path" \
    --output_dir "$output_path" \
    --train_in_8bit False \
    --use_lora False \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --gradient_checkpointing True \
    --global_batch_size 4 \
    --per_device_batch_size 1 \
    --num_epochs 5
