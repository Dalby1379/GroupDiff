# Standard library imports
import argparse
import datetime
import logging
import math
import os
import sys
import time

# Third-party imports
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from tqdm import trange

# Local imports
from models.group_utils import reshape_group_to_batch
from utils.builders import create_generation_model
from utils.misc import ckpt_resume
from utils.train_utils import setup, evaluate_FID
import utils.distributed as distributed

logger = logging.getLogger("GroupDiff")

# performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def pros_process_samples(latents, tokenizer):
    """
    post process the latent to PIL images

    latents: (n, c, h, w)
    tokenizer: tokenizer module
    return: list of PIL images
    """

    latents = 1 / tokenizer.config.scaling_factor * latents
    decode_result = tokenizer.decode(latents)  
    samples = getattr(decode_result, "sample", decode_result) 

    samples = (samples / 2 + 0.5).clamp(0, 1) # [-1, 1] range to [0, 1] range
    samples = samples.cpu().permute(0, 2, 3, 1).float().numpy() * 255.0
    samples = samples.astype(np.uint8)
    samples = [Image.fromarray(sample) for sample in samples]
    return samples

def create_npz_from_sample_folder(sample_dir, num=50_000, num_classes=1000): 
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
            logger.error(
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
def main(args: argparse.Namespace) -> int:
    """
    Run evaluation.
    """
    accelerator = Accelerator()
    distributed.set_accelerator(accelerator)
    
    global logger
    args.accelerator = True  # Enable accelerator mode
    wandb_logger = setup(args, accelerator)


    if args.cond_group_size is None:
        args.cond_group_size = 1

    if args.uncond_group_size is None:
        args.uncond_group_size = args.num_max_sample


    eval_group_size = max(args.cond_group_size, args.uncond_group_size)

    # Setup DDP
    rank = accelerator.process_index if accelerator else 0
    world_size = accelerator.num_processes if accelerator else 1
    device = rank % torch.cuda.device_count()

    seed = args.seed * world_size + rank
    torch.manual_seed(seed)

    # initialize models
    model, tokenizer, ema_model = create_generation_model(args)

    # resume from checkpoint if needed
    if args.auto_resume:
        logger.info("Auto-resume enabled")
        ckpt_resume(args, model, None, None, ema_model)
        

    if args.use_ema and ema_model is not None:
        ema_model.store(model)
        ema_model.copy_to(model)

    # Move model and VAE to device
    model = model.to(device=device, dtype=torch.bfloat16)
    if tokenizer is not None:
        tokenizer = tokenizer.to(device=device, dtype=torch.bfloat16)
        tokenizer.eval()
    model = model.eval()  # important!

    # Create eval directory following the same pattern as evaluate_generator
    cfg = args.cfg
    use_ema = args.use_ema
    eval_dir = f"{args.eval_dir}/epoch_{args.start_epoch:03d}_use_ema={use_ema}-cfg={cfg}"
    eval_start_time = time.perf_counter()
    
    if rank == 0:
        os.makedirs(eval_dir, exist_ok=True)
    
    # Wait for rank 0 to create the directory before other processes try to write to it
    accelerator.wait_for_everyone()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.eval_bsz
    global_batch_size = n * world_size
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    # Ensure total_samples_to_generate is a multiple of (global_batch_size / group_size) * group_size
    # This ensures that we can form complete groups of samples with the same class ID.
    num_classes = 1000  # Assuming 1000 classes for ImageNet, adjust if different

    # Calculate total groups needed across all classes to meet num_images
    groups_per_class_target = math.ceil(args.num_images / num_classes / eval_group_size)
    total_groups_to_generate = groups_per_class_target * num_classes

    # Total samples considering the group size and distribution across GPUs
    total_samples_to_generate = total_groups_to_generate * eval_group_size
    # Adjust total_samples_to_generate to be a multiple of global_batch_size for even distribution
    if total_samples_to_generate % global_batch_size != 0:
        total_samples_to_generate = math.ceil(total_samples_to_generate / global_batch_size) * global_batch_size

    if rank == 0:
        logger.info(f"Total number of images that will be sampled: {total_samples_to_generate}")
    
    assert total_samples_to_generate % world_size == 0, "total_samples_to_generate must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples_to_generate // world_size)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    
    iterations = int(samples_needed_this_gpu // n)
    total = 0  # Keeps track of the total number of samples generated globally so far
    
    logger.info("Generating images for evaluation...")
    n_sampling_steps = args.num_sampling_steps
    temperature = args.temperature
    num_iter = args.num_iter
    logger.info(
        f"Setting: {use_ema=}, {cfg=}, {n_sampling_steps=}, {num_iter=} num_images={args.num_images}, {temperature=}, {total_samples_to_generate=}"
    )
    gen_time, save_time, gen_cnt = 0, 0, 0
    gen_start = time.perf_counter()

    # Unified Sampling: Pre-generate class labels for each group
    # We need to generate class IDs for each 'group' (which consists of config.inference.group_size samples)
    # The total number of 'groups' to generate is total_samples_to_generate // config.inference.group_size
    num_groups_to_generate = total_samples_to_generate // eval_group_size
    all_group_class_ids = []
    for i in range(num_groups_to_generate):
        all_group_class_ids.append(i % num_classes)  # Cycle through classes for each group

    # Convert to a torch tensor for easier slicing. This tensor holds the class ID for each *group*.
    all_group_class_ids_tensor = torch.tensor(all_group_class_ids, device=device)

    # Calculate the number of groups processed per global batch
    groups_per_global_batch = global_batch_size // eval_group_size
    groups_per_gpu_batch = n // eval_group_size
    
    if rank == 0:
        iter_range = trange(iterations, desc=f"Rank{rank}", position=rank)
    else:
        iter_range = range(iterations)
    
    for cur_idx in iter_range:
        # Calculate the starting index for groups for the current global batch
        current_global_group_start_idx = total // eval_group_size

        # Get the slice of group class IDs for the current global batch
        current_global_batch_group_ids = all_group_class_ids_tensor[
            current_global_group_start_idx : current_global_group_start_idx + groups_per_global_batch
        ]

        # Distribute these group class IDs to the current GPU's batch
        start_group_idx_on_gpu = rank * groups_per_gpu_batch
        end_group_idx_on_gpu = start_group_idx_on_gpu + groups_per_gpu_batch
        current_gpu_group_ids = current_global_batch_group_ids[start_group_idx_on_gpu:end_group_idx_on_gpu]

        # 'y' needs to be (n / group_size) x group_size, where each row has the same class ID.
        # current_gpu_group_ids has shape (n / group_size,)
        # We need to unsqueeze and repeat to get the desired shape.
        y = current_gpu_group_ids.unsqueeze(1).repeat(1, eval_group_size)
        assert y.shape == (int(n / eval_group_size), eval_group_size)

        # Generate samples
        start_time = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            samples = model.generate(
                n_samples=n,
                labels=reshape_group_to_batch(y),
                cfg=args.cfg,
                args=args,
            )
            samples = pros_process_samples(samples, tokenizer)
        gen_time += time.perf_counter() - start_time
        gen_cnt += len(samples)
        
        img_per_gpu_per_sec = gen_cnt / gen_time
        elapsed_time = time.perf_counter() - gen_start
        eta = elapsed_time / (cur_idx + 1) * (iterations - cur_idx - 1)
        logger.info(
            f"[{cur_idx+1}/{iterations}] Generated {gen_cnt} images in {gen_time:.2f}s. "
            f"Images per second per gpu: {img_per_gpu_per_sec:.4f}. "
            f"Seconds per image: {gen_time / gen_cnt:.4f}. "
            f"Elapsed time: {str(datetime.timedelta(seconds=elapsed_time))} "
            f"ETA (save time included): {str(datetime.timedelta(seconds=eta))}"
        )
        logger.info(f"FIDs will be logged to {args.log_dir}/eval_summary.txt")
        
        # Save samples to disk as individual .png files, including class label in filename
        start_time = time.perf_counter()
        for i in range(n):  # Iterate through each sample generated on this GPU in the current batch
            global_sample_index = (total + rank * n) + i  # Correct global index for this sample
            # Only save if within the target num_images
            # The class label for this sample comes from the 'y' tensor.
            # y has shape (n/group_size, group_size).
            # The class for sample 'i' is y[i // eval_group_size, 0].
            sample_class_label = y[i // eval_group_size, 0].item()
            samples[i].save(f"{eval_dir}/{global_sample_index:06d}_class-{sample_class_label:04d}.png")
        
        total += global_batch_size  # Update global sample count
        save_time += time.perf_counter() - start_time
        del samples
        torch.cuda.empty_cache()

    # synchronize across processes
    accelerator.wait_for_everyone()
    
    if rank == 0:
        num_imgs = len(os.listdir(eval_dir))
        # sanity check to make sure the number of images is correct
        logger.info(f"Final number of images: {num_imgs}")
    
    # restore EMA parameters (if used)
    if use_ema and ema_model is not None:
        ema_model.restore(model)
    
    accelerator.wait_for_everyone()
    
    # Evaluate FID on rank 0
    if rank == 0:
        metrics_dict = evaluate_FID(eval_dir, None, fid_stats_path=args.fid_stats_path, num_images=args.num_images, num_classes=args.num_classes)
        fid = metrics_dict["frechet_inception_distance"]
        inception_score = metrics_dict["inception_score_mean"]
        
        # Save eval summary in the same format as evaluate_generator
        log_str = f"Epoch {args.start_epoch}, {use_ema=}, {cfg=}, guidance_low={args.guidance_low}, guidance_high={args.guidance_high}, seed={args.seed}, num_sampling_steps={n_sampling_steps}, num_images={args.num_images}, fid={fid}, is={inception_score}"
        with open(f"{args.log_dir}/eval_summary.txt", "a") as f:
            f.write(log_str + "\n")
    
    # ensure evaluation is done before cleanup
    accelerator.wait_for_everyone()
    
    # distributed cleanup (following evaluate_generator pattern)
    if not args.keep_eval_folder:
        cleanup_start_time = time.perf_counter()
        # each GPU removes only its own files (only up to num_images)
        subset_files = []
        for fn in os.listdir(eval_dir):
            if len(fn) < 6 or not fn[:6].isdigit():
                continue
            idx = int(fn[:6])
            if idx >= args.num_images:
                continue
            # Determine which rank produced this index under global-batch slicing
            if (idx % global_batch_size) // n == rank:
                subset_files.append(os.path.join(eval_dir, fn))
        for file_path in subset_files:
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
    
        accelerator.wait_for_everyone()
        
        # rank 0 removes the directories if they are empty
        if rank == 0:
            if not os.listdir(eval_dir):
                os.rmdir(eval_dir)
            logger.info(f"Removed evaluation folder: {eval_dir}")
        logger.info(f"Cleanup time: {time.perf_counter() - cleanup_start_time:.2f}s")
    
    accelerator.wait_for_everyone()
    
    # ensure all processes wait here before proceeding
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    time_str = str(datetime.timedelta(seconds=time.perf_counter() - eval_start_time))
    logger.info(f"Total evaluation time (gen+save+cleanup): {time_str}")
    logger.info(f"Results saved in {args.log_dir}/eval_summary.txt")
    
    return 0


def get_args_parser():
    parser = argparse.ArgumentParser("Generation model evaluation", add_help=False)

    # basic parameters
    parser.add_argument("--start_epoch", default=0, type=int)

    # model parameters
    parser.add_argument("--model", default="DiT_xl", type=str)
    parser.add_argument("--patch_size", default=1, type=int)
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument("--force_one_d_seq", type=int, default=0, help="1d tokens, e.g., 128 for MAETok")
    parser.add_argument("--legacy_mode", action="store_true")
    parser.add_argument("--num_max_sample", default=1, type=int, help="Max numbers in the group")
    parser.add_argument("--cond_group_size", default=1, type=int, help="conditional model group size")
    parser.add_argument("--uncond_group_size", default=1, type=int, help="unconditional model group size")


    # tokenizer parameters
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--tokenizer", default="sdvae", type=str)
    parser.add_argument("--token_channels", default=4, type=int)
    parser.add_argument("--tokenizer_patch_size", default=8, type=int)

    # logging parameters
    parser.add_argument("--output_dir", default="./work_dirs")

    # checkpoint parameters
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--load_from", type=str, default=None, help="load from pretrained model")

    # evaluation parameters
    parser.add_argument("--num_images", default=50000, type=int)
    parser.add_argument("--fid_stats_path", type=str, default="data/fid_stats/adm_in256_stats.npz")
    parser.add_argument("--keep_eval_folder", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_bsz", type=int, default=256)


    # generation parameters
    parser.add_argument("--num_iter", default=64, type=int, help="number of autoregressive steps for MAR")
    parser.add_argument("--noise_schedule", type=str, default="linear", help="noise schedule for diffusion")
    parser.add_argument("--cfg", default=4.0, type=float, help="cfg value for diffusion")
    parser.add_argument("--guidance_low", default=0.0, type=float, help="lower bound for timesteps to apply CFG (0-1 for SiT, 0-1000 for DiT)")
    parser.add_argument("--guidance_high", default=1.0, type=float, help="upper bound for timesteps to apply CFG (0-1 for SiT, 0-1000 for DiT)")

    # diffusion parameters
    parser.add_argument("--num_sampling_steps", type=str, default="250")
    parser.add_argument("--temperature", default=1.0, type=float)

    # dataset parameters
    parser.add_argument("--num_classes", default=1000, type=int)
    
    # EMA parameter
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model for evaluation")
    parser.set_defaults(use_ema=True)
    
    # system parameters
    parser.add_argument("--seed", default=0, type=int)

    # wandb parameters
    parser.add_argument("--project", default="GroupDiff", type=str)
    parser.add_argument("--entity", default="YOUR_WANDB_ENTITY", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")


    # used on train 
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--ema_rate", default=0.9999, type=float)
    parser.add_argument("--resume_from", default=None, help="resume model weights and optimizer state")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exit_code = main(args)
    sys.exit(exit_code)
