# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------


"""
Feature extraction script for ImageNet dataset.
Extracts features from multiple models (VAE, CLIP, SigLIP, DINOv2) and creates metadata.json.
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
import json
import os
import sys
from tqdm import tqdm

from diffusers.models import AutoencoderKL
from transformers import (
    CLIPModel, CLIPProcessor,
    SiglipModel, SiglipProcessor,
    AutoModel, AutoImageProcessor,
    Dinov2Model, AutoProcessor
)

# Enable TF32 for faster training on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#################################################################################
#                             Model Configurations                              #
#################################################################################

MODEL_REGISTRY = {
    "clip-b": {
        "type": "clip",
        "hub_id": "openai/clip-vit-base-patch32",
        "description": "CLIP ViT-Base/32"
    },
    "clip-l": {
        "type": "clip",
        "hub_id": "openai/clip-vit-large-patch14",
        "description": "CLIP ViT-Large/14"
    },
    "siglip": {
        "type": "siglip",
        "hub_id": "google/siglip2-large-patch16-256",
        "description": "SigLIP 2 Large/16 (256px)"
    },
    "dinov2-b": {
        "type": "dinov2",
        "hub_id": "facebook/dinov2-base",
        "description": "DINOv2 Base"
    },
    "dinov2-l": {
        "type": "dinov2",
        "hub_id": "facebook/dinov2-large",
        "description": "DINOv2 Large"
    }
}

#################################################################################
#                             Helper Functions                                  #
#################################################################################

def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class FeatureExtractor:
    def __init__(self, model_name, device):
        self.name = model_name
        self.device = device
        self.config = MODEL_REGISTRY[model_name]
        self.model_type = self.config["type"]

        print(f"Loading {model_name} ({self.config['hub_id']})...")

        if self.model_type == "clip":
            self.model = CLIPModel.from_pretrained(self.config["hub_id"]).to(device)
            self.processor = CLIPProcessor.from_pretrained(self.config["hub_id"])
        elif self.model_type == "siglip":
            self.model = AutoModel.from_pretrained(self.config["hub_id"], device_map=device).eval()
            self.processor = AutoProcessor.from_pretrained(self.config["hub_id"])
        elif self.model_type == "dinov2":
            self.model = Dinov2Model.from_pretrained(self.config["hub_id"]).to(device)
            self.processor = AutoImageProcessor.from_pretrained(self.config["hub_id"])

        self.model.eval()

    @torch.no_grad()
    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        # Prepare inputs
        if self.model_type == "clip":
            inputs = self.processor(text=[""], images=images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            # Use image_embeds and normalize
            embeds = outputs.image_embeds
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=-1)

        elif self.model_type == "siglip":
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            # SigLIP features are already often normalized or used as is, but standard practice for similarity is normalization
            embeds = outputs
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=-1)

        elif self.model_type == "dinov2":
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # Use pooler_output (CLS token)
            embeds = outputs.pooler_output
            # Normalize DINOv2 features? Usually yes for retrieval/clustering.
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return embeds.cpu().numpy()


#################################################################################
#                                  Main Extraction Loop                         #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Feature extraction requires at least one GPU."

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Parse requested models
    requested_models = args.models
    if "all" in requested_models:
        requested_models = list(MODEL_REGISTRY.keys()) + ["vae-256"]

    # Filter valid models
    active_models = []
    use_vae = False

    for m in requested_models:
        if m == "vae-256":
            use_vae = True
        elif m in MODEL_REGISTRY:
            active_models.append(m)
        else:
            if rank == 0:
                print(f"Warning: Model {m} not found in registry. Skipping.")

    # Setup output directories
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        os.makedirs(os.path.join(args.features_path, "labels"), exist_ok=True)

        if use_vae:
            os.makedirs(os.path.join(args.features_path, "vae-256"), exist_ok=True)

        for m in active_models:
            os.makedirs(os.path.join(args.features_path, m), exist_ok=True)

    dist.barrier()

    # Load VAE if requested
    vae = None
    if use_vae:
        assert args.image_size % 8 == 0, "Image size must be divisible by 8 for VAE."
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae_type}").to(device)
        vae.eval()

    # Load other models
    extractors = {}
    for m in active_models:
        extractors[m] = FeatureExtractor(m, device)

    # Setup data
    # We use a custom transform that returns the PIL image (cropped) + Tensor (normalized for VAE)
    # Actually, simpler to just return the PIL image and process it per model
    # BUT VAE needs Tensor.
    # Let's return (PIL_Image, Tensor_for_VAE)

    class CustomTransform:
        def __init__(self, image_size, flip):
            self.image_size = image_size
            self.flip = flip
            self.to_tensor = transforms.ToTensor()
            self.normalize_vae = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        def __call__(self, img):
            # Center crop
            img = center_crop_arr(img, self.image_size)

            # Flip
            if self.flip and torch.rand(1) < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Prepare VAE input: Tensor [-1, 1]
            tensor = self.to_tensor(img)
            tensor_vae = self.normalize_vae(tensor)

            # Return PIL for other models, Tensor for VAE
            return img, tensor_vae

    dataset = ImageFolder(
        args.data_path,
        transform=CustomTransform(args.image_size, args.random_flip)
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )


    def collate_fn(batch):
        # batch is list of ((pil, tensor), label)
        pils = [x[0][0] for x in batch]
        tensors = torch.stack([x[0][1] for x in batch])
        labels = torch.tensor([x[1] for x in batch])
        return (pils, tensors), labels

    loader = DataLoader(
        dataset,
        batch_size=1, # Keep batch size 1 for simplicity with varied model inputs
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )

    # Extract features
    train_steps = 0
    metadata_entries = []
    pbar = None

    if rank == 0:
        pbar = tqdm(total=len(loader), desc="Extracting features", unit="sample")

    for (pil_imgs, vae_tensors), labels in loader:
        # Batch size is 1
        pil_img = pil_imgs[0] # PIL Image
        vae_tensor = vae_tensors.to(device) # (1, 3, H, W)
        label = labels.to(device)

        sample_meta = {
            "index": train_steps,
            "rank": rank,
            "label": int(label.item()),
            "features": {}
        }

        # 1. VAE Extraction
        if use_vae:
            with torch.no_grad():
                latent = vae.encode(vae_tensor).latent_dist.sample().mul_(0.18215)
            latent_np = latent.detach().cpu().numpy()

            save_path = f"{args.features_path}/vae-256/{train_steps}_rank_{rank}.npy"
            np.save(save_path, latent_np)
            sample_meta["features"]["vae-256"] = {
                "shape": list(latent_np.shape),
                "path": os.path.basename(save_path) # relative to features root implies structure
            }

        # 2. Other Models Extraction
        # We pass the PIL image list [pil_img]
        for name, extractor in extractors.items():
            embeds = extractor([pil_img]) # returns numpy (1, D)

            save_path = f"{args.features_path}/{name}/{train_steps}_rank_{rank}.npy"
            np.save(save_path, embeds)

            sample_meta["features"][name] = {
                "shape": list(embeds.shape),
                "path": os.path.basename(save_path)
            }

        # Save Label
        label_np = label.detach().cpu().numpy()
        np.save(f"{args.features_path}/labels/{train_steps}_rank_{rank}.npy", label_np)

        metadata_entries.append(sample_meta)
        train_steps += 1

        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    # Collect metadata
    dist.barrier()
    all_metadata = [None] * world_size
    dist.all_gather_object(all_metadata, metadata_entries)

    if rank == 0:
        flat_metadata = []
        for rank_data in all_metadata:
            if rank_data:
                flat_metadata.extend(rank_data)

        flat_metadata.sort(key=lambda x: (x["rank"], x["index"]))

        metadata = {
            "total_samples": len(flat_metadata),
            "num_ranks": world_size,
            "image_size": args.image_size,
            "models": list(active_models) + (["vae-256"] if use_vae else []),
            "class_to_idx": dataset.class_to_idx,
            "idx_to_class": {v: k for k, v in dataset.class_to_idx.items()},
            "samples": flat_metadata
        }

        metadata_path = os.path.join(args.features_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Extraction complete! Metadata saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from ImageNet")
    parser.add_argument("--data-path", type=str, default="/mnt/localssd/imagenet_raw/train")
    parser.add_argument("--features-path", type=str, default="/mnt/localssd/imagenet_feats/train")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--models", nargs="+", default=["vae-256", "clip-b", "clip-l", "siglip", "dinov2-b", "dinov2-l"],
                        help="List of models to extract. Options: vae-256, clip-b, clip-l, siglip, dinov2-b, dinov2-l, all")
    parser.add_argument("--vae-type", type=str, default="ema", choices=["ema", "mse"])
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--random-flip", action="store_true")

    args = parser.parse_args()
    main(args)
