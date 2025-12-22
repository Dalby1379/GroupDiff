# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from einops import rearrange
from torch import Tensor


def reshape_by_group(x: Tensor, group_size: int) -> Tensor:
    """
    Reshape a tensor, split the batch to N samples in each group.
    """
    x = reshape_group_to_batch(x)
    x = reshape_batch_to_group(x, group_size)
    return x


def reshape_batch_to_group(x: Tensor, group_size: int) -> Tensor:
    """
    Reshape a tensor, split the batch to N samples in each group.
    """
    if x.shape[0] % group_size != 0:
        raise ValueError(f"Batch size {x.shape[0]} is not divisible by group size {group_size}")
    x = rearrange(x, "(b g) ... -> b g ...", b=x.shape[0] // group_size, g=group_size)
    return x


def reshape_group_to_batch(x: Tensor) -> Tensor:
    """
    Reshape a group of tensors to a batch of tensors.
    """
    return rearrange(x, "b g ... -> (b g)...")
