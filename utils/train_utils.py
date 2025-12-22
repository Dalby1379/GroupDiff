# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Adopted from DeTok
#     DeTok: https://github.com/Jiawei-Yang/DeTok/blob/main/utils/train_utils.py

# Standard library imports
import argparse
import datetime
import json
import logging
import os
import time
from functools import partial

# Third-party imports
import numpy as np
import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from PIL import Image, ImageFile
from torch import Tensor
from tqdm import tqdm

# Local imports
import utils.distributed as dist
import utils.misc as misc
from models.group_utils import reshape_group_to_batch
from utils.logger import MetricLogger, SmoothedValue, WandbLogger, setup_logging, setup_wandb
from utils.fid_evaluator import main as fid_evaluator_main

tqdm = partial(tqdm, dynamic_ncols=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger("GroupDiff")


def setup(args: argparse.Namespace, accelerator: Accelerator | None = None):
    """setup distributed training, logging, and experiment configuration"""
    enable_accelerator = getattr(args, "accelerator", False)
    if not enable_accelerator:
        dist.enable_distributed()
    global logger

    if args.exp_name is None:
        args.exp_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_exp"

    base_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    args.log_dir = base_dir
    args.ckpt_dir = os.path.join(base_dir, "checkpoints")
    args.vis_dir = os.path.join(base_dir, "visualization")
    args.eval_dir = os.path.join(base_dir, "eval")

    if not enable_accelerator:
        global_rank, world_size = dist.get_global_rank(), dist.get_world_size()
    else:
        global_rank, world_size = accelerator.process_index, accelerator.num_processes
    args.world_size = world_size
    args.global_bsz = args.batch_size * world_size
    args.print_freq = 100 if args.global_bsz < 512 else args.print_freq

    # misc.fix_random_seeds(args.seed + global_rank)

    args.warmup_epochs = int(getattr(args, "warmup_rate", 0) * args.epochs)

    wandb_logger = None
    if global_rank == 0:
        for path in [args.log_dir, args.ckpt_dir, args.vis_dir, args.eval_dir]:
            os.makedirs(path, exist_ok=True)

        if args.enable_wandb and not enable_accelerator:
            wandb_logger = setup_wandb(
                args=args,
                entity=args.entity,
                project=args.project,
                name=args.exp_name,
                log_dir=args.log_dir,
            )

        setup_logging(output=args.log_dir, name="GroupDiff", rank0_log_only=True)
        logger.info(f"Logging to {args.log_dir}")
        json_config = json.dumps(args.__dict__, indent=4, sort_keys=True)
        logger.info(json_config)

        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        json_path = os.path.join(args.log_dir, f"args_{time_str}.json")
        with open(json_path, "w") as f:
            json.dump(args.__dict__, f, indent=4)
        logger.info(f"Args saved to {json_path}")

    if getattr(args, "use_aligned_schedule", False):
        args.grad_clip = 0
        args.weight_decay = 0
        args.lr = 0.0002
        args.warmup_epochs = 0

    # Auto-enable load_latent when use_cached_tokens is enabled
    if getattr(args, "use_cached_tokens", False):
        if not getattr(args, "load_latent", False):
            args.load_latent = True
            if global_rank == 0:
                logger.info("Auto-enabling load_latent because use_cached_tokens is True")

    tokenizer = getattr(args, "tokenizer", None)
    if tokenizer:
        token_channels_map = {"vavae": 32, "maetok-b-128": 32, "sdvae": 4, "eqvae": 4}
        args.token_channels = token_channels_map.get(tokenizer, args.token_channels)
    return wandb_logger


def train_one_epoch_generator(
    args: argparse.Namespace,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler: misc.NativeScalerWithGradNormCount,
    wandb_logger: WandbLogger | None,
    epoch: int,
    ema_model: torch.nn.Module,
    tokenizer: torch.nn.Module | None = None,
):
    model.train(True)
    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file, prefetch=True)
    metric_logger.add_meter("lr", SmoothedValue(1, "{value:.6f}"))
    metric_logger.add_meter("samples/s/gpu", SmoothedValue(args.print_freq, "{avg:.2f}"))
    steps_per_epoch = len(data_loader)
    header = f"Epoch: [{epoch}]"
    logger.info(f"log dir: {args.log_dir}")
    start_time = time.perf_counter()

    for step, data_dict in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        # calibrate 1 epoch = 1000 iterations regardless of batch size
        frac_epoch = step / steps_per_epoch + epoch  # fraction of the current epoch
        calib_global_step = int(frac_epoch * 1000)
        tokenization_time = 0.0
        group_size = 1
        if args.use_cached_tokens:
            # load posterior moments and sample
            latents, labels = data_dict["latents"], data_dict["labels"]
            x = latents
            if x.ndim == 5:
                if labels.ndim == 2:
                    group_size = labels.shape[1]
                    labels = reshape_group_to_batch(labels)

                x = reshape_group_to_batch(x)

            # decode the latents to images to debug
            if step == 0:
                num_debug = min(4, x.shape[0])
                decoded_latents = 1 / tokenizer.config.scaling_factor * x[:num_debug]
                decoded = tokenizer.decode(decoded_latents)
                decoded_imgs = decoded.sample
                decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
                decoded_imgs = (decoded_imgs * 255).cpu().permute(0, 2, 3, 1).float().numpy()
                for i in range(num_debug):
                    img_arr = decoded_imgs[i].astype(np.uint8)
                    pil_img = Image.fromarray(img_arr)
                    pil_img.save(os.path.join(args.log_dir, f"debug_decoded_latents_{i}.png"))

            # x = tokenizer.sample_from_moments(moments)
        elif args.tokenizer is not None:
            # online tokenization
            imgs, labels = data_dict["pixel_values"], data_dict["labels"]
            # tokenization time estimate is not strictly accurate, but it's a good approximation
            tokenizer_start_time = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if imgs.ndim == 5:
                    group_size = imgs.shape[0]
                    imgs = reshape_group_to_batch(imgs)
                    labels = reshape_group_to_batch(labels)
                x = tokenizer.encode(imgs).latent_dist.sample().mul_(0.18215)

                # decode the latents to images to debug
                if step == 0:
                    num_debug = min(4, x.shape[0])
                    decoded_latents = 1 / tokenizer.config.scaling_factor * x[:num_debug]
                    decoded = tokenizer.decode(decoded_latents)
                    decoded_imgs = decoded.sample
                    decoded_imgs = (decoded_imgs / 2 + 0.5).clamp(0, 1)
                    decoded_imgs = (decoded_imgs * 255).cpu().permute(0, 2, 3, 1).float().numpy()
                    for i in range(num_debug):
                        img_arr = decoded_imgs[i].astype(np.uint8)
                        pil_img = Image.fromarray(img_arr)
                        pil_img.save(os.path.join(args.log_dir, f"debug_decoded_latents_{i}.png"))
            # Save the first image(s) for visualization
            if step == 0:
                num_vis = min(4, imgs.shape[0]) if isinstance(imgs, torch.Tensor) else 1
                vis_imgs = imgs[:num_vis] if isinstance(imgs, torch.Tensor) else imgs
                if isinstance(vis_imgs, torch.Tensor):
                    # Un-normalize if needed and convert to uint8
                    vis_imgs_np = vis_imgs.clone().detach().cpu()
                    if vis_imgs_np.dtype == torch.float32 or vis_imgs_np.dtype == torch.float16 or vis_imgs_np.dtype == torch.bfloat16:
                        # Assume pixel range is [-1, 1]
                        vis_imgs_np = vis_imgs_np * 0.5 + 0.5
                        vis_imgs_np = vis_imgs_np * 255.0
                    vis_imgs_np = vis_imgs_np.clamp(0, 255).to(torch.uint8)
                    if vis_imgs_np.ndim == 4:
                        for i in range(num_vis):
                            img = vis_imgs_np[i]
                            img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
                            pil_img = Image.fromarray(img)
                            pil_img.save(os.path.join(args.log_dir, f"debug_first_img_{i}.png"))
                    elif vis_imgs_np.ndim == 3:
                        img = vis_imgs_np.permute(1, 2, 0).numpy()
                        pil_img = Image.fromarray(img)
                        pil_img.save(os.path.join(args.log_dir, "debug_first_img.png"))
                else:
                    logger.info("Could not visualize images; imgs is not a Tensor.")
            tokenization_time = time.perf_counter() - tokenizer_start_time

        else:
            # pixel-space inputs, good luck : )
            x, labels = data_dict["img"], data_dict["label"]

        misc.adjust_learning_rate(optimizer, frac_epoch, args)

        # forward pass
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if group_size > 1:
                drop_tokens = True
            else:
                drop_tokens = False
            # print("group_size", group_size, "drop_tokens", drop_tokens)
            loss = model(x, labels, group_size=group_size, drop_tokens=drop_tokens, timesteps_offset=args.timesteps_offset)
            loss_value = loss.item()

        # backward pass
        grad_norm = loss_scaler(loss, optimizer, args.grad_clip, model.parameters())
        optimizer.zero_grad(set_to_none=True)

        # update ema model
        ema_model.step(model)

        torch.cuda.synchronize()

        # log metrics
        loss_value_reduced = dist.all_reduce_mean(loss_value)
        psnr = -10 * np.log10(loss_value_reduced)
        samples_per_second_per_gpu = args.batch_size * (step + 1) / (time.perf_counter() - start_time)
        samples_per_second = samples_per_second_per_gpu * args.world_size
        metric_logger.update(
            loss=loss_value_reduced,
            psnr=psnr,
            grad_norm=grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            tokenization=tokenization_time,
            **{"samples/s/gpu": samples_per_second_per_gpu, "samples/s": samples_per_second},
        )
        if wandb_logger is not None and step % args.print_freq == 0:
            log_dict = {
                "psnr": psnr,
                "loss": loss_value_reduced,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm,
                "tokenization": tokenization_time,
                "samples_per_sec_per_gpu": samples_per_second_per_gpu,
                "samples_per_sec": samples_per_second,
            }
            wandb_logger.update(log_dict, step=calib_global_step)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_img_save_format(grid, max_pixels=2_000_000):
    grid_height, grid_width = grid.shape[-2:]
    total_pixels = grid_height * grid_width
    return "jpg" if total_pixels > max_pixels else "png"


@torch.inference_mode()
def to_uint8_numpy(tensor: Tensor) -> np.ndarray:
    return (tensor * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

@torch.inference_mode()
def visualize_generator(
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    epoch: int,
    use_emas: list[bool] = [True],
):
    model.eval()
    if args.class_of_interest is not None:
        assert all(0 <= c < args.num_classes for c in args.class_of_interest)
        class_labels = torch.tensor(args.class_of_interest, device="cuda", dtype=torch.long)
    else:
        class_labels = torch.randint(args.num_classes, (8,), device="cuda")

    n_samples = len(class_labels)

    for use_ema in use_emas:
        if use_ema and ema_model is not None:
            ema_model.store(model)
            ema_model.copy_to(model)
        elif use_ema and ema_model is None:
            logger.warning("use_ema is True but ema_model is None, skipping EMA.")
            continue

        for cfg in [args.cfg, 1.0]:
            logger.info(f"Generating images with cfg={cfg}, n_imgs={n_samples}, ema={use_ema}")
            generated_images = generate_images(args, model, tokenizer, labels=class_labels, cfg=cfg)
            generated_images = dist.concat_all_gather(generated_images).cpu()

            if dist.is_main_process():
                grid = torchvision.utils.make_grid(generated_images, n_samples, 8, pad_value=1)
                format = get_img_save_format(grid)
                outpath = os.path.join(args.vis_dir, f"ep{epoch:04d}_cfg={cfg}_ema={use_ema}.{format}")
                torchvision.utils.save_image(grid, outpath)
                logger.info(f"Saved at {outpath}")

            accelerator = dist.get_accelerator()
            if accelerator:
                accelerator.wait_for_everyone()
            elif dist.is_enabled():
                torch.distributed.barrier()
            torch.cuda.empty_cache()

        if use_ema and ema_model is not None:
            ema_model.restore(model)

    accelerator = dist.get_accelerator()
    if accelerator:
        accelerator.wait_for_everyone()
    elif dist.is_enabled():
        torch.distributed.barrier()
    torch.cuda.empty_cache()

@torch.inference_mode()
def generate_images(
    args: argparse.Namespace,
    generator: torch.nn.Module,
    tokenizer: torch.nn.Module | None,
    labels: list[int] | Tensor,
    cfg: float = 1.0,
): # return generated images in [0, 1] range in numpy array
    if not isinstance(labels, Tensor):
        labels = torch.tensor(labels, dtype=torch.long).to("cuda")
    # generator = generator.eval().to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated = generator.generate(n_samples=len(labels), cfg=cfg, labels=labels, args=args)
        if tokenizer is not None:
            # generated = tokenizer.detokenize(generated)
            # Decode using VAE
            generated = 1 / tokenizer.config.scaling_factor * generated
            generated = tokenizer.decode(generated).sample
            generated = (generated / 2 + 0.5).clamp(0, 1)
            generated = generated.cpu().permute(0, 2, 3, 1).float().numpy()
            generated = DiffusionPipeline.numpy_to_pil(generated)

    return generated


def create_npz_from_sample_folder(sample_dir, num=50_000, num_classes=1000):  # Added num_classes
    """
    Builds a single .npz file from a folder of .png samples, with unified sampling per class.
    """
    samples = []
    samples_per_class_count = {i: 0 for i in range(num_classes)}
    samples_per_class_target = math.ceil(num / num_classes)  # Calculate target samples per class

    # Get all image files and sort them to maintain consistency
    all_image_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".png")])

    pbar = tqdm(total=num, desc="Building .npz file from samples with unified sampling")
    for filename in all_image_files:
        if len(samples) >= num:  # Stop if we've collected enough total samples
            break

        # Assuming filename format is "XXXXXX_class-YYY.png"
        try:
            # Extract class label from filename
            class_label = int(filename.split("_class-")[1].replace(".png", ""))
        except (IndexError, ValueError):
            logger.warning(
                f"Warning: Could not extract class label from {filename}. Skipping for unified sampling."
            )
            continue

        if samples_per_class_count[class_label] < samples_per_class_target:
            sample_pil = Image.open(os.path.join(sample_dir, filename))
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples.append(sample_np)
            samples_per_class_count[class_label] += 1
            pbar.update(1)  # Update progress bar for each sample added

    pbar.close()
    samples = np.stack(samples)
    # Assert the final shape matches the requested number of samples
    assert samples.shape[0] == num, f"Expected {num} samples, but got {samples.shape[0]}."
    assert samples.shape[1] == samples.shape[1], "Height mismatch."  # Original logic, no change
    assert samples.shape[2] == samples.shape[2], "Width mismatch."  # Original logic, no change
    assert samples.shape[3] == 3, "Channel mismatch."  # Original logic, no change

    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

@torch.inference_mode()
def evaluate_FID(
    save_folder: str,
    reference_folder: str | None = None,
    prc: bool = False,
    fid_stats_path: str | None = None,
    num_images: int = 50_000,
    num_classes: int = 1000,
):
    """
    Evaluate FID using the custom fid_evaluator.

    Args:
        save_folder: folder containing generated images as .png files
        reference_folder: not used, kept for compatibility (uses fid_stats_path instead)
        prc: whether to compute precision/recall (always computed in new evaluator)
        fid_stats_path: path to reference statistics .npz file

    Returns:
        dict with keys: frechet_inception_distance, inception_score_mean, sfid, precision, recall
    """
    logger.info("Calculating FID for %s...", save_folder)

    # Create NPZ file from the sample folder with per-class uniform sampling
    sample_npz_path = create_npz_from_sample_folder(save_folder, num=num_images, num_classes=num_classes)

    # Use the fid_stats_path as reference
    assert fid_stats_path is not None, "fid_stats_path must be provided for FID evaluation"

    # Call the FID evaluator
    results = fid_evaluator_main(
        ref_batch=fid_stats_path,
        sample_batch=sample_npz_path,
    )

    # Map results to expected format for backward compatibility
    metrics_dict = {
        "frechet_inception_distance": results["FID"],
        "inception_score_mean": results["Inception Score"],
        "sfid": results["sFID"],
        "precision": results["Precision"],
        "recall": results["Recall"],
    }

    fid = metrics_dict["frechet_inception_distance"]
    inception_score = metrics_dict["inception_score_mean"]
    logger.info("Folder: %s", save_folder)
    logger.info("Metrics: %s", metrics_dict)
    logger.info("FID: %.4f, IS: %.4f", fid, inception_score)

    # Clean up the temporary NPZ file
    try:
        os.remove(sample_npz_path)
        logger.info("Removed temporary NPZ file: %s", sample_npz_path)
    except Exception as e:
        logger.warning("Failed to remove temporary NPZ file %s: %s", sample_npz_path, e)

    return metrics_dict