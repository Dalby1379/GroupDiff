#!/bin/bash

# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Evaluation the pretrained GroupDiff-l-4 (SiT-XL) model.

project=GroupDiff-l-pretrained
exp_name=gdiff-l-4-sit-xl-2-repa-resume
checkpoint_path=released_model/${exp_name}.pth
batch_size=32  # per GPU batch size
YOUR_WANDB_ENTITY="YourWandbEntity"

# Evaluate the groupdiff model
accelerate launch --num_processes 8 --multi_gpu --mixed_precision=bf16 eval.py \
    --project $project --exp_name $exp_name --auto_resume \
    --model SiT_xl --patch_size 2 --num_max_sample 4 \
    --batch_size $batch_size --eval_bsz 128 \
    --num_sampling_steps 250 --cfg 2.585 \
    --guidance_low 0.25 --guidance_high 0.75 \
    --cond_group_size 1 --uncond_group_size 4 \
    --num_images 50000 --seed 0 \
    --load_from ${checkpoint_path} --use_ema \
    --fid_stats_path data/VIRTUAL_imagenet256_labeled.npz \
    --entity $YOUR_WANDB_ENTITY
