# Copyright © 2025, Adobe Inc. and its licensors. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

# Modified from DeTok
#     DeTok: https://github.com/Jiawei-Yang/DeTok/blob/main/utils/distributed.py

import functools
import os
import random
import re
import socket
import sys

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn

_LOCAL_RANK = -1
_LOCAL_WORLD_SIZE = -1

_TORCH_DISTRIBUTED_ENV_VARS = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
)

# Global accelerator instance for Hugging Face Accelerate
_accelerator = None


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        if isinstance(x, torch.Tensor):
            x_reduce = x.clone().detach().cuda()
        else:
            x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce = x_reduce.float() / world_size
        return x_reduce.item()
    return x


def concat_all_gather(tensor, gather_dim=0) -> torch.Tensor:
    if dist.get_world_size() == 1:
        return tensor
    output = dist_nn.functional.all_gather(tensor)
    return torch.cat(output, dim=gather_dim)


def is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    return dist.get_rank() if is_enabled() else 0


def get_world_size():
    return dist.get_world_size() if is_enabled() else 1


def is_main_process() -> bool:
    return get_global_rank() == 0


def _is_slurm_job_process() -> bool:
    return "SLURM_JOB_ID" in os.environ and not os.isatty(sys.stdout.fileno())


def _parse_slurm_node_list(s: str) -> list[str]:
    nodes = []
    p = re.compile(r"(([^\[]+)(?:\[([^\]]+)\])?),?")
    for m in p.finditer(s):
        prefix, suffixes = s[m.start(2) : m.end(2)], s[m.start(3) : m.end(3)]
        for suffix in suffixes.split(","):
            span = suffix.split("-")
            if len(span) == 1:
                nodes.append(prefix + suffix)
            else:
                width = len(span[0])
                start, end = int(span[0]), int(span[1]) + 1
                nodes.extend([prefix + f"{i:0{width}}" for i in range(start, end)])
    return nodes


def _get_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@functools.lru_cache
def enable_distributed():
    if _is_slurm_job_process():
        os.environ["MASTER_ADDR"] = _parse_slurm_node_list(os.environ["SLURM_JOB_NODELIST"])[0]
        os.environ["MASTER_PORT"] = str(random.Random(os.environ["SLURM_JOB_ID"]).randint(20_000, 60_000))
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["LOCAL_WORLD_SIZE"] = str(
            int(os.environ["WORLD_SIZE"]) // int(os.environ["SLURM_JOB_NUM_NODES"])
        )
    elif "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_get_available_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])


def set_accelerator(accelerator):
    """
    Set global accelerator instance.
    
    Args:
        accelerator: Hugging Face Accelerator instance
    """
    global _accelerator
    _accelerator = accelerator


def get_accelerator():
    """
    Get global accelerator instance.
    
    Returns:
        The accelerator instance or None if not set
    """
    return _accelerator


def all_reduce_mean_accelerator(x, accelerator=None):
    """
    All-reduce mean using accelerator if available, otherwise use torch.distributed.
    
    Args:
        x: Value to reduce (Tensor or scalar)
        accelerator: Optional accelerator instance. If None, uses global _accelerator
        
    Returns:
        Reduced mean value
    """
    acc = accelerator or _accelerator
    if acc is not None:
        if isinstance(x, torch.Tensor):
            x_reduce = x.clone().detach()
        else:
            x_reduce = torch.tensor(x, device=acc.device)
        x_reduce = acc.reduce(x_reduce, reduction="mean")
        return x_reduce.item() if not isinstance(x, torch.Tensor) else x_reduce
    else:
        return all_reduce_mean(x)


def concat_all_gather_accelerator(tensor, gather_dim=0, accelerator=None):
    """
    Concatenate all gather using accelerator if available.
    
    Args:
        tensor: Tensor to gather
        gather_dim: Dimension to concatenate along
        accelerator: Optional accelerator instance. If None, uses global _accelerator
        
    Returns:
        Gathered and concatenated tensor
    """
    acc = accelerator or _accelerator
    if acc is not None:
        if acc.num_processes == 1:
            return tensor
        gathered = acc.gather(tensor)
        return gathered
    else:
        return concat_all_gather(tensor, gather_dim)


def get_rank_accelerator(accelerator=None):
    """
    Get rank using accelerator if available.
    
    Args:
        accelerator: Optional accelerator instance. If None, uses global _accelerator
        
    Returns:
        Process rank
    """
    acc = accelerator or _accelerator
    if acc is not None:
        return acc.process_index
    else:
        return get_global_rank()


def get_world_size_accelerator(accelerator=None):
    """
    Get world size using accelerator if available.
    
    Args:
        accelerator: Optional accelerator instance. If None, uses global _accelerator
        
    Returns:
        Number of processes
    """
    acc = accelerator or _accelerator
    if acc is not None:
        return acc.num_processes
    else:
        return get_world_size()


def is_main_process_accelerator(accelerator=None):
    """
    Check if this is the main process using accelerator if available.
    
    Args:
        accelerator: Optional accelerator instance. If None, uses global _accelerator
        
    Returns:
        True if this is the main process
    """
    acc = accelerator or _accelerator
    if acc is not None:
        return acc.is_main_process
    else:
        return is_main_process()


def barrier_accelerator(accelerator=None):
    """
    Synchronization barrier using accelerator if available.
    
    Args:
        accelerator: Optional accelerator instance. If None, uses global _accelerator
    """
    acc = accelerator or _accelerator
    if acc is not None:
        acc.wait_for_everyone()
    elif is_enabled():
        dist.barrier()