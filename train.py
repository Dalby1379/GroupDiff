# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Modified from DeTok
#     DeTok: https://github.com/Jiawei-Yang/DeTok/blob/main/train.py

"""
GroupDiff: Generation model training script.
"""

import argparse
import datetime
import logging
import sys
import time

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms

import utils.distributed as distributed
from utils.builders import create_generation_model, create_optimizer_and_scaler
from utils.misc import ckpt_resume, save_checkpoint
from utils.train_utils import (
    setup,
    train_one_epoch_generator,
)
from dataset.cluster_data import ImageNetDataset

# performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

logger = logging.getLogger("GroupDiff")


def dict_collate_fn(batch):
    """
    Collate function that returns a dictionary compatible with MetaDataLoader injection.
    Handles new dictionary format from dataset: {"img": ..., "label": ..., "latent": ..., "feats": ...}
    """
    # Extract images and labels from dict format
    imgs = []
    labels = []
    latents = []
    feats_list = []

    for sample in batch:
        if isinstance(sample, dict):
            # New dict format
            if sample.get("img") is not None:
                imgs.append(sample["img"])
            labels.append(sample["label"])
            if sample.get("latent") is not None:
                latents.append(sample["latent"])
            if sample.get("feats") is not None:
                feats_list.append(sample["feats"])
        else:
            # Fallback for old tuple format (backward compatibility)
            img, label = sample
            imgs.append(img)
            labels.append(label)

    # Stack/convert to tensors
    result = {}
    if imgs:
        result["pixel_values"] = torch.stack(imgs)
    result["labels"] = torch.tensor(labels)

    # Add latents if present (these will be used instead of pixel_values if load_latent=True)
    if latents:
        stacked_latents = torch.stack(latents)
        # Remove extra dimension if latents have shape [B, 1, C, H, W] -> [B, C, H, W]
        if stacked_latents.ndim == 5 and stacked_latents.shape[1] == 1:
            stacked_latents = stacked_latents.squeeze(1)
        result["latents"] = stacked_latents

    # Add features if present
    if feats_list:
        result["feats"] = feats_list

    return result


def main(args: argparse.Namespace) -> int:
    # Initialize accelerator
    accelerator = Accelerator()
    distributed.set_accelerator(accelerator)
    
    global logger
    args.accelerator = True  # Enable accelerator mode
    wandb_logger = setup(args, accelerator)

    if args.cond_group_size is None:
        args.cond_group_size = 1

    if args.uncond_group_size is None:
        args.uncond_group_size = args.num_max_sample

    seed = args.seed * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)

    # initialize models
    model, tokenizer, ema_model = create_generation_model(args)
    optimizer, loss_scaler = create_optimizer_and_scaler(args, model)
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    model_wo_ddp = model.module if accelerator.num_processes > 1 else model

    # resume from checkpoint if needed
    if args.auto_resume:
        logger.info("Auto-resume enabled")
        ckpt_resume(args, model_wo_ddp, optimizer, loss_scaler, ema_model)

    # Setup image transforms (following train_example.py pattern)
    # Only create transforms if we're loading images
    transform_train = None
    if not args.load_latent or args.load_image:
        transform_train = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # Create dataset (following train_example.py pattern)
    logger.info("Initializing dataset...")
    logger.info(f"Loading mode: load_image={args.load_image}, load_latent={args.load_latent}")

    # Determine latent_root: defaults to features_root, then to data_path
    latent_root = args.latent_root if args.latent_root is not None else (args.features_root if args.features_root is not None else args.data_path)
    logger.info(f"Latent root directory: {latent_root}")

    # Calculate batch sizes for epoch planning
    # Use per-GPU batch size for planning, so that after partition each GPU gets correct number of iterations
    # Example: 1.28M samples / 256 per-gpu-batch = 5000 batches -> partition to 8 GPUs = 625 batches per GPU
    world_size = accelerator.num_processes

    dataset_kwargs = {
        "root_dir": args.data_path,
        "image_size": args.img_size,
        "seq_length": args.num_max_sample,
        "global_batch_size": args.batch_size,  # Use per-GPU batch size for planning
        "use_epoch_planning": (args.num_max_sample > 1),
        "group_ratio": args.group_ratio,
        "num_workers": args.num_workers,
        "load_image": args.load_image,
        "load_latent": args.load_latent,
        "latent_feature_name": args.latent_feature_name,
        "latent_root": latent_root,
        "features_root": args.features_root,
    }

    # Add metadata_path if provided
    if hasattr(args, 'metadata_path') and args.metadata_path is not None:
        dataset_kwargs["metadata_path"] = args.metadata_path

    # Add cluster_manager configuration based on query_sim strategy
    if args.query_sim == 'feat' and args.faiss_index_path is not None:
        logger.info("Using FAISS-based (feature) search strategy for epoch planning")
        cluster_manager_config = {
            "target": "dataset.cluster_data.ClusterDataManager",
            "params": {
                "strategy": "faiss",
                "faiss_index_path": args.faiss_index_path,
                "feature_name": args.faiss_feature_name,
                "features_root": args.features_root,
                "num_workers": args.cluster_num_workers,
            }
        }
        dataset_kwargs["cluster_manager"] = cluster_manager_config
    elif args.query_sim == 'class':
        logger.info("Using class-based search strategy for epoch planning")
        cluster_manager_config = {
            "target": "dataset.cluster_data.ClusterDataManager",
            "params": {
                "strategy": "class",
            }
        }
        dataset_kwargs["cluster_manager"] = cluster_manager_config
    else:
        logger.info("No cluster_manager specified, using default class-based strategy")

    logger.info(f"dataset_kwargs: {dataset_kwargs}")
    logger.info(f"Batch size config: per_gpu_batch_size={args.batch_size}, world_size={world_size}")

    dataset = ImageNetDataset(**dataset_kwargs)

    # Set transform manually after instantiation (following train_example.py)
    # Only set if we're loading images
    if transform_train is not None:
        if hasattr(dataset, "set_transform"):
            dataset.set_transform(transform_train)
        else:
            if hasattr(dataset, "transform"):
                dataset.transform = transform_train

    logger.info("Dataset initialized successfully")

    # Get parameters from dataset
    use_epoch_planning = dataset.use_epoch_planning
    cluster_manager = dataset.cluster_manager

    logger.info(f"Dataset created with {len(dataset)} samples")
    logger.info(f"Use epoch planning: {use_epoch_planning}")

    # training loop
    logger.info(f"Start training from {args.start_epoch} to {args.epochs}")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        # --- Epoch Planning Phase ---
        global_epoch_plan = None
        should_use_planning = use_epoch_planning

        if use_epoch_planning and cluster_manager is not None:
            # Generate epoch plan (handles broadcasting in DDP)
            if accelerator.is_main_process:
                logger.info(f"Planning epoch {epoch}...")

            global_epoch_plan = cluster_manager.get_epoch_plan()

            if accelerator.is_main_process and global_epoch_plan:
                sequence_count = sum(1 for batch in global_epoch_plan if isinstance(batch, dict) and batch.get("type") == "sequence")
                direct_count = sum(1 for batch in global_epoch_plan if isinstance(batch, dict) and batch.get("type") == "direct")
                logger.info(f"Created epoch plan with {len(global_epoch_plan)} total batches ({sequence_count} sequence, {direct_count} direct)")

        # Check if planning succeeded
        if global_epoch_plan is None and use_epoch_planning:
            if accelerator.is_main_process:
                logger.warning("Failed to create epoch plan, falling back to traditional mode")
            should_use_planning = False

        # Sync before creating loader (following train_example.py)
        accelerator.wait_for_everyone()

        # --- Create DataLoader with planning ---
        if should_use_planning and global_epoch_plan is not None and cluster_manager is not None:
            logger.info(f"Creating DataLoader for epoch {epoch} with epoch planning")
            logger.info(f"Using epoch planning with {len(global_epoch_plan)} batches")

            # Use epoch planning with DDP sampler
            world_size = accelerator.num_processes
            rank = accelerator.process_index

            sampler = cluster_manager.get_sampler(
                global_epoch_plan,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=42,
                epoch=epoch,
            )

            # Create new loader with planned batches
            planned_loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                collate_fn=dict_collate_fn,
            )

            # Wrap with MetaDataLoader to inject metadata
            data_loader_train = cluster_manager.wrap_dataloader(planned_loader, sampler.local_plan)
        else:
            # Traditional mode (when epoch planning is disabled or num_max_sample == 1)
            data_loader_train = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
                collate_fn=dict_collate_fn,
            )

        train_one_epoch_generator(
            args, model, data_loader_train, optimizer, loss_scaler, wandb_logger,
            epoch, ema_model, tokenizer
        )

        # progress logging
        elapsed_t = time.time() - start_time + args.last_elapsed_time
        eta = elapsed_t / (epoch + 1) * (args.epochs - epoch - 1)
        logger.info(
            f"[{epoch}/{args.epochs}] "
            f"Accumulated elapsed time: {str(datetime.timedelta(seconds=int(elapsed_t)))}, "
            f"ETA: {str(datetime.timedelta(seconds=int(eta)))}"
        )

        # checkpointing
        should_save = (
            (epoch + 1) % args.save_freq == 0  # save every n epochs
            or (epoch + 1) == args.epochs  # save at the end of training
        )

        if should_save:
            save_checkpoint(args, epoch, model_wo_ddp, optimizer, loss_scaler, ema_model, elapsed_t)
            accelerator.wait_for_everyone()

    # training complete
    total_time = int(time.time() - start_time + args.last_elapsed_time)
    logger.info(f"Training time {str(datetime.timedelta(seconds=total_time))}")
    
    return 0


def get_args_parser():
    parser = argparse.ArgumentParser("Generation model training", add_help=False)

    # basic training parameters
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU for training")

    # model parameters
    parser.add_argument("--model", default="MAR_base", type=str)
    parser.add_argument("--patch_size", default=1, type=int)
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument("--force_one_d_seq", type=int, default=0, help="1d tokens, e.g., 128 for MAETok")
    parser.add_argument("--legacy_mode", action="store_true")
    parser.add_argument("--num_max_sample", default=1, type=int, help="Max numbers in the group")
    parser.add_argument("--cond_group_size", default=1, type=int, help="conditional model group size")
    parser.add_argument("--uncond_group_size", default=1, type=int, help="unconditional model group size")
    parser.add_argument("--timesteps_offset", default=0.0, type=float, help="timesteps offset for group diversity")


    # tokenizer parameters
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--tokenizer", default="sdvae", type=str)
    parser.add_argument("--token_channels", default=4, type=int)
    parser.add_argument("--tokenizer_patch_size", default=8, type=int)

    # logging parameters
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--last_elapsed_time", type=float, default=0.0)

    # checkpoint parameters
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_from", default=None, help="resume model weights and optimizer state")
    parser.add_argument("--load_from", type=str, default=None, help="load from pretrained model")
    parser.add_argument("--load_tokenizer_from", type=str, default=None, help="load from pretrained tokenizer")
    parser.add_argument("--keep_n_ckpts", default=1, type=int, help="keep the last n checkpoints")
    parser.add_argument("--milestone_interval", default=20, type=int, help="keep checkpoints every n epochs")


    # optimization parameters
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (used directly without scaling). If not set, uses --blr scaled by batch size")
    parser.add_argument("--blr", type=float, default=1e-4, help="Base learning rate (scaled by batch size if --lr not set)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for cosine schedule")
    parser.add_argument("--lr_sched", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--warmup_rate", type=float, default=0.0, help="warmup_ep = warmup_rate * total_ep")
    parser.add_argument("--ema_rate", default=0.9999, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--use_aligned_schedule", action="store_true")

    # generation parameters
    parser.add_argument("--noise_schedule", type=str, default="linear", help="noise schedule for diffusion")

    # diffusion loss parameters
    parser.add_argument("--num_sampling_steps", type=str, default="100")

    # dataset parameters
    parser.add_argument("--use_cached_tokens", action="store_true")
    parser.add_argument("--data_path", default="./data/imagenet/train", type=str)
    parser.add_argument("--metadata_path", default="data/imagenet_feats/train/metadata.json", type=str, help="Path to metadata.json (auto-detected if None)")
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # data loading mode parameters
    parser.add_argument("--load_image", action="store_true", default=False, help="Load raw images")
    parser.add_argument("--no_load_image", action="store_false", dest="load_image", help="Don't load raw images")
    parser.add_argument("--load_latent", action="store_true", default=True, help="Load VAE latents")
    parser.add_argument("--latent_feature_name", default="vae-256", type=str, help="Name of latent feature in metadata")
    parser.add_argument("--latent_root", default=None, type=str, help="Root directory for latent files (defaults to features_root or data_path)")

    # cluster data / epoch planning parameters
    parser.add_argument("--group_ratio", default=0.1, type=float, help="Ratio of group batches in epoch planning")
    parser.add_argument("--query_sim", default=None, type=str, choices=['class', 'feat'], help="Query similarity strategy: 'class' for class-based search, 'feat' for FAISS-based feature search")
    parser.add_argument("--faiss_index_path", default="data/imagenet_feats/train_index/dinov2-l/imagenet_index_ivfpq.faiss", type=str, help="Path to FAISS index file")
    parser.add_argument("--faiss_feature_name", default="dinov2-l", type=str, help="Feature name for FAISS search")
    parser.add_argument("--features_root", default="data/imagenet_feats/train", type=str, help="Root directory for feature files")
    parser.add_argument("--cluster_num_workers", default=32, type=int, help="Number of threads for cluster query processing")

    # system parameters
    parser.add_argument("--seed", default=0, type=int)

    # wandb parameters
    parser.add_argument("--project", default="GroupDiff", type=str)
    parser.add_argument("--entity", default="YOUR_WANDB_ENTITY", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exit_code = main(args)
    sys.exit(exit_code)