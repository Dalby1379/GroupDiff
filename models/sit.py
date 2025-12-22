# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

# Modified from https://github.com/Jiawei-Yang/DeTok/blob/main/models/sit.py
# Referece DiT: https://github.com/facebookresearch/DiT/blob/main/models.py


# This file implements the latent diffusion model with split class-condition model and unconditional model.
from abc import ABC

import torch
from diffusers.utils.torch_utils import randn_tensor

from transport import Sampler, create_transport

from .denoiser import Denoiser
from .group_utils import reshape_batch_to_group, reshape_group_to_batch
from .model_utils import SIZE_DICT

class SiT(torch.nn.Module, ABC):

    def __init__(self, 
        img_size=256,
        patch_size=1,
        model_size="base",
        tokenizer_patch_size=16,
        token_channels=16,
        label_drop_prob=0.1,
        num_classes=1000,
        num_sampling_steps: int = 250,
        sampling_method="euler",
        grad_checkpointing=False,
        learn_sigma=False,  # no learn_sigma in SiT
        legacy_mode=False,  # also not output sigma in SiT
        num_max_sample: int = 1, # max number of samples to generate
        timesteps_offset: float = 0.0,
    ) -> None:
        super().__init__()
        
        # --------------------------------------------------------------------------
        # basic configuration
        self.token_channels = token_channels
        self.out_channels = token_channels * 2 if learn_sigma else token_channels
        self.input_size = img_size // tokenizer_patch_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.grad_checkpointing = grad_checkpointing
        self.learn_sigma = learn_sigma
        self.legacy_mode = legacy_mode
        self.num_max_sample = num_max_sample
        self.sampling_method = sampling_method
        self.num_sampling_steps = int(num_sampling_steps)

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
        ))

        self.transport = create_transport(
            train_eps=0.0,
            sample_eps=0.0,
        )
        self.sampler = Sampler(self.transport)

        self.sample_fn = self.sampler.sample_sde(
            sampling_method=self.sampling_method,
            num_steps=self.num_sampling_steps,
        )

    def net(self, x, t, y, drop_tokens=False, keep_group_size=False, group_size=1,**kwargs):
        if drop_tokens:
            y = torch.full_like(y, self.num_classes)
        return self.denoiser(x=reshape_batch_to_group(x,group_size=group_size), t=reshape_batch_to_group(t,group_size=group_size), y=reshape_batch_to_group(y,group_size=group_size), group_size=group_size,keep_group_shape=keep_group_size).pred


    def get_timesteps(self, x, timesteps_offset=0.0):
        """
        Sample timesteps for training with optional offset for group diversity.
        
        Args:
            x: Input tensor with shape (batch_size, group_size, ...)
            timesteps_offset: Maximum time offset for group timesteps. If 0, all group members
                            get the same timestep. Otherwise, each member gets a timestep within
                            [base_t - offset, base_t + offset] range, clamped to [0, 1].
        
        Returns:
            Timesteps tensor of shape (batch_size, group_size) with values in [0, 1]
        """
        batch_size = x.shape[0]
        group_size = x.shape[1]
        device = x.device
        
        # Get time interval from transport
        t0, t1 = self.transport.check_interval(self.transport.train_eps, self.transport.sample_eps)
        
        # Sample base timesteps using transport's time distribution
        dist_options = self.transport.time_dist_type.split("_")
        if dist_options[0] == "uniform":
            base_t = torch.rand((batch_size,), device=device) * (t1 - t0) + t0
        elif dist_options[0] == "logit-normal":
            from transport.transport import truncated_logitnormal_sample
            mu, sigma = float(dist_options[1]), float(dist_options[2])
            base_t = truncated_logitnormal_sample((batch_size,), mu, sigma, t0, t1).to(device)
        else:
            raise NotImplementedError(f"Unknown time distribution type {self.transport.time_dist_type}")
        
        # Apply time shift
        base_t = self.transport.time_dist_shift * base_t / (1 + (self.transport.time_dist_shift - 1) * base_t)
        
        if timesteps_offset == 0.0:
            # All group members get the same timestep
            timesteps = base_t.unsqueeze(1).repeat(1, group_size)
        else:
            # Each group member gets a slightly different timestep
            timesteps = torch.zeros((batch_size, group_size), device=device, dtype=torch.float32)
            for b in range(batch_size):
                t_base = base_t[b]
                min_t = max(0.0, t_base - timesteps_offset)
                max_t = min(1.0, t_base + timesteps_offset)
                # Sample uniformly in the range
                group_t = torch.rand((group_size,), device=device) * (max_t - min_t) + min_t
                timesteps[b, :] = group_t
        
        return timesteps

    def forward(self, x, y, group_size=1, drop_tokens=False, timesteps_offset=0.0):
        """
        Forward pass for training with optional group-wise timestep diversity.
        
        Args:
            x: Input tensor
            y: Labels
            group_size: Number of samples per group
            drop_tokens: Whether to drop tokens (for group training)
            timesteps_offset: Time offset for group diversity (0 means all same timestep)
        """
        if group_size > 1:
            # Generate group-wise timesteps
            x_grouped = reshape_batch_to_group(x, group_size)
            t = self.get_timesteps(x_grouped, timesteps_offset)
            # Flatten back to batch dimension
            t = reshape_group_to_batch(t)
        else:
            # For single samples, use default transport sampling
            t = None
        
        model_kwargs = dict(y=y, drop_tokens=drop_tokens, group_size=group_size) if drop_tokens else dict(y=y)
        loss_dict = self.transport.training_losses(self.net, x, model_kwargs=model_kwargs, t=t)
        return loss_dict["loss"].mean()

    def forward_with_cfg(
        self,
        x: torch.Tensor,  # B , C , H , W
        t: torch.Tensor,  # B
        class_labels: torch.Tensor,  # B
        class_null: torch.Tensor,  # B
        guidance_low: float,
        guidance_high: float,
        cfg_scale: float,
        cond_group_size: int,
        uncond_group_size: int,
    ) -> torch.Tensor:
        # x: (B, M, C, H, W)
        current_bs = x.shape[0]  # Use current tensor shape
        latent_input_cond = x
        latent_input_uncond = x

        if t.ndim == 0:
            t_record = t.item()
        else:
            t_record = t[0].item()

        # Build timestep tensors shaped (B, M)
        if t.ndim == 0:
            t = t[None]
        timesteps = t.to(x.device, dtype=torch.float32)
        timesteps = timesteps.expand(current_bs)

        timesteps_cond = timesteps
        timesteps_uncond = timesteps

        if guidance_low <= t_record <= guidance_high:
            apply_guidance = True
        else:
            apply_guidance = False

        cond_pred = self.denoiser(
            x=reshape_batch_to_group(latent_input_cond, group_size=cond_group_size),
            t=reshape_batch_to_group(timesteps_cond, group_size=cond_group_size),
            y=reshape_batch_to_group(class_labels, group_size=cond_group_size),
            group_size=cond_group_size,
            keep_group_shape=False,
        ).pred

        if apply_guidance:
            uncond_pred = self.denoiser(
                x=reshape_batch_to_group(latent_input_uncond, group_size=uncond_group_size),
                t=reshape_batch_to_group(timesteps_uncond, group_size=uncond_group_size),
                y=reshape_batch_to_group(class_null, group_size=uncond_group_size),
                group_size=uncond_group_size,
                keep_group_shape=False,
            ).pred

            # pred shape B, C, H, W
            num_guided_channels = 3 if self.denoiser.config.in_channels >= 3 else self.denoiser.config.in_channels
            eps_cond_guide = cond_pred[:, :num_guided_channels]
            eps_uncond_guide = uncond_pred[:, :num_guided_channels]
            eps_guide = eps_uncond_guide + cfg_scale * (eps_cond_guide - eps_uncond_guide)
            rest_cond = cond_pred[:, num_guided_channels:]
            velocity = torch.cat([eps_guide, rest_cond], dim=1)

        else:
            velocity = cond_pred

        return velocity.to(x.dtype)

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        labels, 
        cfg: float = 1.0,
        args: None = None,
    ):
        """generate samples using the model."""
        device = labels.device

        # Get guidance range (for SiT, timesteps are in [0, 1], no scaling needed)
        guidance_low = getattr(args, 'guidance_low', 0.25) if args is not None else 0.0
        guidance_high = getattr(args, 'guidance_high', 0.75) if args is not None else 1.0

        # Get group sizes with defaults
        cond_group_size = getattr(args, 'cond_group_size', 1) if args is not None else 1
        uncond_group_size = getattr(args, 'uncond_group_size', 1) if args is not None else 1

        latent_size = self.denoiser.config.input_size
        latent_channels = self.denoiser.config.in_channels

        latents = randn_tensor(
            shape=(n_samples, latent_channels, latent_size, latent_size),
            device=device,
            dtype=torch.bfloat16,
        )

        class_null = torch.tensor([self.num_classes] * n_samples, device=device)

        model_fn = self.forward_with_cfg


        model_kwargs = {
            "class_labels": labels,
            "class_null": class_null,
            "guidance_low": guidance_low,
            "guidance_high": guidance_high,
            "cfg_scale": cfg,
            "cond_group_size": cond_group_size,
            "uncond_group_size": uncond_group_size,
        }

        latents = self.sample_fn(
            latents,
            model_fn,
            **model_kwargs,
        )[-1]

        return latents


# model size variants
def SiT_XL(**kwargs) -> SiT:
    return SiT(model_size="xl", **kwargs)


def SiT_L(**kwargs) -> SiT:
    return SiT(model_size="large", **kwargs)


def SiT_B(**kwargs) -> SiT:
    return SiT(model_size="base", **kwargs)


def SiT_S(**kwargs) -> SiT:
    return SiT(model_size="small", **kwargs)



SiT_models = {"SiT_base": SiT_B, "SiT_large": SiT_L, "SiT_xl": SiT_XL, "SiT_small": SiT_S}