# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

"""
Modified from https://github.com/facebookresearch/DiT/blob/main/models.py
"""
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------

import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

from .group_utils import reshape_batch_to_group, reshape_group_to_batch


@dataclass
class DenoiserOutput:
    """DiT output."""

    pred: torch.Tensor
    hidden_states: list[torch.Tensor] | None = None
    attn_maps: list[torch.Tensor] | None = None


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train=False, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention from the patches of one image to its own context tokens (text and image tokens).
    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.kv = nn.Linear(hidden_size, hidden_size * 2, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, context):
        """
        x: (N, T, D) tensor of patch tokens.
        context: (N, L, D) tensor of context tokens.
        """
        n, t, d = x.shape
        head_dim = d // self.num_heads
        q = self.q(x).reshape(n, t, self.num_heads, head_dim).transpose(1, 2)
        k, v = self.kv(context).reshape(n, context.shape[1], 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(n, t, d))


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cross_attention=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        if cross_attention:
            self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads)
        else:
            self.cross_attn = None
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, context=None):
        """
        Forward pass of DiTBlock.
        x: (N, M, T, D) tensor of inputs, where N is batch size, M is number of samples, T is number of patches, and D is hidden size.
        c: (N, M, D) tensor of conditioning inputs, where N is batch size, M is number of samples, and D is hidden size.
        context: optional (N * M, L, D) tensor of context tokens. Each image attends to its own context only.
        """
        n, m, t = x.shape[:3]
        x = reshape_group_to_batch(x)  # (N * M, T, D) # Flatten batch and sample dimensions
        c = reshape_group_to_batch(c)  # (N * M, D) # Flatten batch and sample dimensions
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x_shift = modulate(self.norm1(x), shift_msa, scale_msa)

        x = x + gate_msa.unsqueeze(1) * rearrange(
            self.attn(rearrange(x_shift, "(b m) t d -> b (m t) d", b=n, m=m)), "b (m t) d-> (b m) t d", b=n, m=m, t=t
        )
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        x = reshape_batch_to_group(x, m)  # before returning, we need to reshape x back to (N, M, T, D)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        # x: (N, M, T, D) tensor of inputs, where N is batch size, M is number of samples, T is number of patches, and D is hidden size.
        n, m, t = x.shape[:3]
        x = reshape_group_to_batch(x)  # (N * M, T, D) # Flatten batch and sample dimensions
        c = reshape_group_to_batch(c)  # (N * M, D) # Flatten batch and sample dimensions

        ## Vanilla DiT Final Projection
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        x = reshape_batch_to_group(x, m)  # before returning, we need to reshape x back to (N, M, T, D)
        return x


class Denoiser(nn.Module):
    @dataclass
    class Config:
        in_channels: int = 16
        input_size: int = 32
        patch_size: int = 2
        hidden_size: int = 1152
        depth: int = 28
        num_heads: int = 16
        mlp_ratio: float = 4.0
        num_classes: int = 1000

        learn_sigma: bool = True
        output_sigma: bool = False
        max_group_size: int = 1

        use_grad_checkpoint: bool = True

        # cross-attention conditioning. context_dim == 0 disables it and keeps the class-conditional model unchanged.
        context_dim: int = 0  # feature size of the text tokens, e.g. 768 for CLIP ViT-L/14
        image_embed_dim: int = 0  # size of a pooled image embedding appended as image tokens; 0 disables it
        num_image_tokens: int = 4

    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.config = deepcopy(config)
        in_channels = config.in_channels
        num_classes = config.num_classes
        input_size = config.input_size
        patch_size = config.patch_size
        hidden_size = config.hidden_size
        depth = config.depth
        num_heads = config.num_heads
        mlp_ratio = config.mlp_ratio
        learn_sigma = config.learn_sigma
        max_group_size = config.max_group_size

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.output_sigma = config.output_sigma

        self.max_group_size = max_group_size
        self.use_grad_checkpoint = config.use_grad_checkpoint

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, 0.1)
        if max_group_size == 0:
            self.sample_embedder = None
        else:
            self.sample_embedder = LabelEmbedder(max_group_size, hidden_size, 0)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.use_context = config.context_dim > 0
        self.num_image_tokens = config.num_image_tokens
        if self.use_context:
            self.context_proj = nn.Linear(config.context_dim, hidden_size, bias=True)
        if config.image_embed_dim > 0:
            assert self.use_context, "image tokens are appended to the text tokens, so context_dim must be set"
            # same form as the image projection of IP-Adapter: a linear layer to a few tokens, then LayerNorm
            self.image_proj = nn.Linear(config.image_embed_dim, config.num_image_tokens * hidden_size, bias=True)
            self.image_norm = nn.LayerNorm(hidden_size)
        else:
            self.image_proj = None

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, cross_attention=self.use_context)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        if self.sample_embedder is not None:
            nn.init.normal_(self.sample_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out the output projection of cross-attention, so that the block starts as the block without it:
        for block in self.blocks:
            if block.cross_attn is not None:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def _generate_random_sample_ids(self, batch_size: int, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate random unique sample IDs for each sequence in the batch.

        Args:
            batch_size: Number of sequences in the batch
            num_samples: Number of samples per sequence (must be <= max_group_size)
            device: Device to create tensors on

        Returns:
            Tensor of shape (batch_size, num_samples) containing unique random sample IDs
            for each sequence from the range [0, max_group_size)
        """
        assert num_samples <= self.max_group_size, (
            f"num_samples ({num_samples}) must be <= max_group_size ({self.max_group_size})"
        )

        # Generate all permutations at once and take first num_samples
        all_permutations = torch.stack([torch.randperm(self.max_group_size, device=device) for _ in range(batch_size)])

        sample_ids = all_permutations[:, :num_samples]
        # breakpoint()
        return sample_ids
        # return all_permutations[:, :num_samples]

    def _generate_fix_sample_ids(self, batch_size: int, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate fixed sample IDs for each sequence in the batch.
        """
        # breakpoint()
        if num_samples == 1:
            ids = (
                torch.arange(0, self.max_group_size, device=device)
                .unsqueeze(1)
                .repeat(batch_size // self.max_group_size, 1)
            ).flatten(0, 1)
            return ids.unsqueeze(1).repeat(1, num_samples)
        else:
            return torch.arange(0, num_samples, device=device).unsqueeze(1).repeat(batch_size, 1)

    def forward(
        self,
        x,
        t,
        y,
        return_hidden_states: bool = False,
        use_fix_sample_ids: bool = False,
        group_size: int = 1,
        keep_group_shape: bool = True,
        context=None,
        image_embeds=None,
    ):
        """
        Forward pass of DiT.
        x: (N, M, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N, M,) tensor of diffusion timesteps
        y: (N, M,) tensor of class labels
        context: optional (N * M, L, context_dim) tensor of text token features, one sequence per image
        image_embeds: optional (N * M, image_embed_dim) tensor of pooled image embeddings, one per image
        """
        if x.ndim == 5:
            bs, group_size, *rest = x.shape

        if x.ndim == 4 and t.ndim == 1 and y.ndim == 1:
            pass

        else:
            assert x.ndim == 5 and t.ndim == 2 and y.ndim == 2, (
                f"Expected x: (N, M, C, H, W), t: (N, M), y: (N, M), got x: {x.shape}, t: {t.shape}, y: {y.shape}"
            )
            x = reshape_group_to_batch(x)
            t = reshape_group_to_batch(t)
            y = reshape_group_to_batch(y)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, train=False)  # (N, D)

        if self.sample_embedder is not None:
            if use_fix_sample_ids:
                sample_ids = self._generate_fix_sample_ids(int(x.shape[0] / group_size ), group_size, x.device)
            else:
                sample_ids = self._generate_random_sample_ids(int(x.shape[0] / group_size ), group_size, x.device)
            sample_ids = rearrange(sample_ids, "b m -> (b m)")  # (N * M, )
            s = self.sample_embedder(sample_ids, False)
        else:
            s = torch.zeros_like(y, device=x.device)

        c = t + y + s  # (N * M, D)

        c = reshape_batch_to_group(c, group_size)
        x = reshape_batch_to_group(x, group_size)

        if self.use_context and context is not None:
            context = self.context_proj(context)  # (N * M, L, D)
            if self.image_proj is not None:
                image_tokens = self.image_proj(image_embeds).reshape(context.shape[0], self.num_image_tokens, -1)
                context = torch.cat([context, self.image_norm(image_tokens)], dim=1)  # (N * M, L + K, D)
        else:
            context = None

        hidden_states = []

        for block in self.blocks:
            if self.use_grad_checkpoint:
                block_inputs = (x, c) if context is None else (x, c, context)
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), *block_inputs, use_reentrant=True)  # (N, T, D)
            else:
                x = block(x, c, context)  # (N, T, D)
            if return_hidden_states:
                hidden_states.append(x)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(reshape_group_to_batch(x))  # (N, out_channels, H, W)

        if keep_group_shape:
            x = reshape_batch_to_group(x, group_size)  # (N, M, out_channels, H, W)

        if self.learn_sigma and not self.output_sigma:
            x = x[:, :, : int(self.out_channels / 2)]

        if return_hidden_states:
            return DenoiserOutput(pred=x, hidden_states=hidden_states)
        else:
            return DenoiserOutput(pred=x)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
