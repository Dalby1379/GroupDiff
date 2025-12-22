# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

# Modified from https://github.com/Jiawei-Yang/DeTok/blob/main/models/dit.py
# Referece DiT: https://github.com/facebookresearch/DiT/blob/main/models.py


import logging

import torch
import torch.nn as nn

from diffusion import create_diffusion

from .denoiser import Denoiser
from .group_utils import reshape_batch_to_group, reshape_group_to_batch
from .model_utils import SIZE_DICT

logger = logging.getLogger("GroupDiff")


class DiT(nn.Module):
    """diffusion model with a transformer backbone."""

    def __init__(
        self,
        img_size=256,
        patch_size=1,
        model_size="base",
        tokenizer_patch_size=16,
        token_channels=16,
        num_classes=1000,
        learn_sigma=True,
        noise_schedule="linear",
        num_sampling_steps=250,
        grad_checkpointing=False,
        legacy_mode=False,
        num_max_sample: int = 1, # max number of samples to generate
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # basic configuration
        self.learn_sigma = learn_sigma
        self.token_channels = token_channels
        self.out_channels = token_channels * 2 if learn_sigma else token_channels
        self.input_size = img_size // tokenizer_patch_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.grad_checkpointing = grad_checkpointing
        self.legacy_mode = legacy_mode
        self.num_max_sample = num_max_sample

        # model architecture configuration
        size_dict = SIZE_DICT[model_size]
        num_layers, num_heads, width = size_dict["layers"], size_dict["heads"], size_dict["width"]

        self.denoiser = Denoiser(config=Denoiser.Config(
            in_channels=token_channels,
            input_size=self.input_size,
            patch_size=self.patch_size,
            hidden_size=width,
            depth=num_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
            use_grad_checkpoint=grad_checkpointing,
            max_group_size=self.num_max_sample,
            output_sigma=learn_sigma,
        ))

        # --------------------------------------------------------------------------
        # diffusion setup
        self.train_diffusion = create_diffusion("", noise_schedule=noise_schedule)
        self.gen_diffusion = create_diffusion(num_sampling_steps, noise_schedule=noise_schedule)

        # log model info
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(
            f"[DiT] params: {num_trainable_params:.2f}M size: {model_size}, num_layers: {num_layers}, width: {width}"
        )

    def unpatchify(self, x):
        """convert patch tokens back to image tensor."""
        group_size = x.shape[1]
        x = reshape_group_to_batch(x)
        c, p = self.out_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        imgs = reshape_batch_to_group(imgs, group_size)
        return imgs

    def generate_fixed_sample_ids(self, batch_size: int, group_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate fixed sample IDs for each sequence in the batch.
        """
        if  group_size == 1:
            ids = (
                torch.arange(0, self.num_max_sample, device=device)
                .unsqueeze(1)
                .repeat(batch_size // self.num_max_sample, 1)
            ).flatten(0, 1)
            return ids.unsqueeze(1).repeat(1, group_size)
        else:
            return torch.arange(0, group_size, device=device).unsqueeze(1).repeat(batch_size, 1)


    def net(self, x, t, y, drop_tokens=False, keep_group_size=False, group_size=1,**kwargs):
        if drop_tokens:
            y = torch.full_like(y, self.num_classes)
        return self.denoiser(x=reshape_batch_to_group(x,group_size=group_size), t=reshape_batch_to_group(t,group_size=group_size), y=reshape_batch_to_group(y,group_size=group_size), group_size=group_size,keep_group_shape=keep_group_size).pred

    def forward_with_cfg(self, x, t, y, cfg_scale, cond_group_size=1, uncond_group_size=1, guidance_low=0.0, guidance_high=1.0):
        """forward pass with classifier-free guidance."""
        half = x[: len(x) // 2]
        cond_y = y[:len(x) // 2]
        uncond_y = y[len(x) // 2:]
        t = t[:len(x) // 2]

        if guidance_low > t[0].item() or t[0].item() > guidance_high:
            # Outside guidance range, only use conditional model
            cond_model_out = self.net(half, t, cond_y, group_size=cond_group_size)
            # Duplicate for batch consistency (CFG expects doubled batch)
            model_out = torch.cat([cond_model_out, cond_model_out], dim=0)
            return model_out
        else:
            cond_model_out = self.denoiser(x=reshape_batch_to_group(half,group_size=cond_group_size), t=reshape_batch_to_group(t,group_size=cond_group_size), y=reshape_batch_to_group(cond_y,group_size=cond_group_size), group_size=cond_group_size,keep_group_shape=False).pred
            uncond_model_out = self.denoiser(x=reshape_batch_to_group(half,group_size=uncond_group_size), t=reshape_batch_to_group(t,group_size=uncond_group_size), y=reshape_batch_to_group(uncond_y,group_size=uncond_group_size), group_size=uncond_group_size,keep_group_shape=False).pred

            if self.legacy_mode:
                cond_eps, cond_rest = cond_model_out[:, :3], cond_model_out[:, 3:]
                uncond_eps, uncond_rest = uncond_model_out[:, :3], uncond_model_out[:, 3:]
            else:
                cond_eps, cond_rest = cond_model_out[:, : self.token_channels], cond_model_out[:, self.token_channels :]
                uncond_eps, uncond_rest = uncond_model_out[:, : self.token_channels], uncond_model_out[:, self.token_channels :]
            # Apply CFG only to epsilon, keep sigma from conditional branch
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            # For learned sigma, only use conditional prediction (not interpolated)
            if self.learn_sigma:
                rest = torch.cat([cond_rest, cond_rest], dim=0)
                return torch.cat([eps, rest], dim=1)
            else:
                return eps

    def get_timesteps(self, x, timesteps_offset=0):
        group_size = x.shape[1]
        timesteps_offset = int(timesteps_offset * self.train_diffusion.num_timesteps)
        t = torch.randint(0, self.train_diffusion.num_timesteps, (x.shape[0],), device=x.device)
        if timesteps_offset  == 0:
            timesteps = t.unsqueeze(1).repeat(1, group_size)
        else:
            timesteps = torch.zeros((x.shape[0], group_size), device=x.device, dtype=torch.long)
            for b in range(x.shape[0]):
                base_t = t[b]
                min_t = max(0, base_t - timesteps_offset)
                max_t = min(self.train_diffusion.num_timesteps - 1, base_t + timesteps_offset)
                group_t = torch.randint(min_t, max_t + 1, (group_size,), device=x.device)
                timesteps[b, :] = group_t
        return timesteps

    def forward(self, x, y, group_size=1, drop_tokens=False, timesteps_offset=0):
        """forward pass for training."""
        t = self.get_timesteps(reshape_batch_to_group(x, group_size), timesteps_offset)
        t = reshape_group_to_batch(t)
        loss_dict = self.train_diffusion.training_losses(self.net, x, t, dict(y=y, drop_tokens=drop_tokens, group_size=group_size))
        return loss_dict["loss"].mean()


    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=1.0, args=None):
        """generate samples using the model."""
        device = labels.device

        # Get guidance range (for DiT, timesteps are in [0, 1000])
        guidance_low = getattr(args, 'guidance_low', 0.0) * 1000.0 if args is not None else 0.0
        guidance_high = getattr(args, 'guidance_high', 1.0) * 1000.0 if args is not None else 1000.0

        # Get group sizes with defaults
        cond_group_size = getattr(args, 'cond_group_size', 1) if args is not None else 1
        uncond_group_size = getattr(args, 'uncond_group_size', 1) if args is not None else 1


        z = torch.randn(n_samples, self.token_channels, self.input_size, self.input_size)
        z = z.to(device)

        # setup classifier-free guidance with dynamic guidance range
        if cfg > 1.0:
            z = torch.cat([z, z], 0)
            labels = torch.cat([labels, torch.full_like(labels, self.num_classes)], 0)
            model_kwargs = dict(y=labels, cfg_scale=cfg, cond_group_size=cond_group_size, uncond_group_size=uncond_group_size, guidance_low=guidance_low, guidance_high=guidance_high)
            sample_fn = self.forward_with_cfg
        else:
            model_kwargs = dict(y=labels, cond_group_size=cond_group_size, uncond_group_size=uncond_group_size)
            sample_fn = self.net

        # generate samples
        samples = self.gen_diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )

        if cfg > 1.0:
            samples, _ = samples.chunk(2, dim=0)  # remove null class samples
        return samples


# model size variants
def DiT_base(**kwargs):
    return DiT(model_size="base", **kwargs)


def DiT_large(**kwargs):
    return DiT(model_size="large", **kwargs)


def DiT_xl(**kwargs):
    return DiT(model_size="xl", **kwargs)


def DiT_huge(**kwargs):
    return DiT(model_size="huge", **kwargs)


DiT_models = {"DiT_base": DiT_base, "DiT_large": DiT_large, "DiT_xl": DiT_xl, "DiT_huge": DiT_huge}

