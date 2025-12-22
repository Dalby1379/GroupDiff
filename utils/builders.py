# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

# Modified and Adopted from DeTok
#     DeTok: https://github.com/Jiawei-Yang/DeTok/blob/main/utils/builders.py
# Modifications:
# - remove non-group related code
# - add support for GroupDiff models
# - set the default VAE to SD-VAE

import logging

import torch.utils.data
import torchvision.transforms as transforms

import models
import utils.distributed as distributed
import utils.losses as losses
from utils.loader import ListDataset, center_crop_arr
from utils.misc import NativeScalerWithGradNormCount
from diffusers.models import AutoencoderKL

logger = logging.getLogger("GroupDiff")

def create_generation_model(args):
    logger.info("Creating generation models.")
    if args.tokenizer is not None:
        if args.tokenizer == "sdvae":
            tokenizer = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        else:
            raise ValueError(f"Unsupported tokenizer {args.tokenizer}")
    else:
        tokenizer = None
        
    if tokenizer is not None:
        tokenizer.cuda()
        tokenizer.eval()
        tokenizer.requires_grad_(False)
        logger.info("====VAE=====")
        logger.info(tokenizer)

    if args.model in models.DiT_models:
        model = models.DiT_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            tokenizer_patch_size=args.tokenizer_patch_size,
            token_channels=args.token_channels,
            num_classes=args.num_classes,
            num_sampling_steps=args.num_sampling_steps,
            grad_checkpointing=args.grad_checkpointing,
            legacy_mode=args.legacy_mode, # legacy mode: cfg on the first three channels only
            num_max_sample=args.num_max_sample,
            noise_schedule=args.noise_schedule,
        )
    elif args.model in models.SiT_models:
        model = models.SiT_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            tokenizer_patch_size=args.tokenizer_patch_size,
            token_channels=args.token_channels,
            num_classes=args.num_classes,
            num_sampling_steps=args.num_sampling_steps,
            grad_checkpointing=args.grad_checkpointing,
            legacy_mode=args.legacy_mode, # legacy mode: cfg on the first three channels only
            num_max_sample=args.num_max_sample,
        )
    else:
        raise ValueError(f"Unsupported model {args.model}")

    model.cuda()
    logger.info("====Model=====")
    logger.info(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Parameters: {n_params / 1e6:.2f}M ({n_params:,})")

    # ema model
    ema = models.SimpleEMAModel(model, decay=args.ema_rate)
    return model, tokenizer, ema


def create_reconstruction_model(args):
    logger.info("Creating reconstruction models.")
    if args.model in models.VAE_models:
        model = models.VAE_models[args.model](
            load_ckpt=not getattr(args, "no_load_ckpt", False),
            gamma=args.gamma,
        )
    elif args.model in models.GroupDiff_models:
        model = models.GroupDiff_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            token_channels=args.token_channels,
            mask_ratio=args.mask_ratio,
            gamma=args.gamma,
        )
    else:
        raise ValueError(f"Unsupported model {args.model}")

    model.cuda()
    logger.info("====Model=====")
    logger.info(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Trainable Parameters: {n_params / 1e6:.2f}M ({n_params:,})")
    ema = models.SimpleEMAModel(model, decay=args.ema_rate)
    return model, ema


def create_optimizer_and_scaler(args, model, print_trainable_params=False):
    logger.info("creating optimizers")

    # exclude parameters from weight decay
    exclude = lambda name, p: (
        p.ndim < 2 or any(keyword in name for keyword in
        ["ln", "bias", "embedding", "norm", "gamma", "embed", "token", "diffloss"])
    )

    named_parameters = list(model.named_parameters())
    no_decay_list = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if not exclude(n, p) and p.requires_grad]
    eff_batch_size = args.batch_size * args.world_size

    # Use lr directly if provided, otherwise fall back to scaled blr
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
        logger.info(f"Using scaled learning rate from blr: base lr = {args.blr:.6e}, effective batch size = {eff_batch_size}")
    else:
        logger.info(f"Using direct learning rate (no scaling): lr = {args.lr:.6e}")

    logger.info(f"actual lr: {args.lr:.6e}")
    logger.info(f"effective batch size: {eff_batch_size}")
    logger.info(f"training with {args.world_size} gpus")
    logger.info(f"weight_decay: {args.weight_decay} on {len(rest_params)} weight tensors")
    logger.info(f"no_decay: {len(no_decay_list)} weight tensors")

    optimizer = torch.optim.AdamW(
        [
            {"params": no_decay_list, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.weight_decay},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    logger.info(f"Optimizer = {str(optimizer)}")
    if print_trainable_params:
        logger.info("trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"\t{name}")

    loss_scaler = NativeScalerWithGradNormCount()
    logger.info(f"Loss Scaler = {str(loss_scaler)}")
    return optimizer, loss_scaler


def create_loss_module(args):
    loss_module = losses.ReconstructionLoss(
        discriminator_start_epoch=getattr(args, "discriminator_start_epoch", 20),
        perceptual_loss=getattr(args, "perceptual_loss", "lpips-convnext_s-1.0-0.1"),
        perceptual_weight=getattr(args, "perceptual_weight", 1.1),
        kl_weight=args.kl_loss_weight,
    )
    loss_module.cuda()
    logger.info("====Loss Module=====")
    # logger.info(loss_module)
    return loss_module
