project=GroupDiff-l
exp_name=groupdiff-l-4-dit-b-dinov2-v2
batch_size=32  # per GPU batch size, global batch size = batch_size x num_gpus = 32 x 8 = 256
epochs=40
YOUR_WANDB_ENTITY="YourWandbEntity"

# Train the groupdiff model
accelerate launch --num_processes 8 --multi_gpu --mixed_precision=bf16 train.py \
    --project $project --exp_name $exp_name --auto_resume \
    --model DiT_base --patch_size 2 --num_max_sample 4 \
    --batch_size $batch_size --epochs $epochs \
    --lr 1e-4 --num_sampling_steps 250 \
    --data_path data/imagenet/train \
    --query_sim feat --use_cached_tokens \
    --entity $YOUR_WANDB_ENTITY

# Evaluate the groupdiff model

checkpoint_path=work_dirs/${project}/${exp_name}/checkpoints/lastest.pth
accelerate launch --num_processes 8 --multi_gpu --mixed_precision=bf16 eval.py \
    --project $project --exp_name $exp_name --auto_resume \
    --model DiT_base --patch_size 2 --num_max_sample 4 \
    --batch_size $batch_size --eval_bsz 128 \
    --num_sampling_steps 250 --cfg 2.5 \
    --guidance_low 0.0 --guidance_high 1.0 \
    --cond_group_size 1 --uncond_group_size 4 \
    --num_images 50000 --seed 0 \
    --load_from ${checkpoint_path} --use_ema \
    --fid_stats_path data/VIRTUAL_imagenet256_labeled.npz \
    --entity $YOUR_WANDB_ENTITY