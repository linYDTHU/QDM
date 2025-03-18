"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]                        # B x half
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = th.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return th.flatten(emb, -2, -1)

@th.no_grad()
def bounding_boxes_embedding(boxes, dim):
    """
    Create bounding boxes embeddings.

    :param boxes: a List of [L_i, 4] tensors.
    :param dim: the dimension of the output.
    :return: a List of [L_i, D] Tensor of positional embeddings.
    """
    # Each coordinate is embedded separately.
    one_fourth_D = dim // 8
    freqs = th.exp(
        -math.log(10000) * th.arange(start=0, end=one_fourth_D, dtype=th.float32)/one_fourth_D
    ).to(device=boxes.device)
    boxes_embedding = []
    for j in range(4):
        args = boxes[:, [j]].float() * freqs[None]
        boxes_embedding.append(get_emb(args))
    boxes_embedding = th.cat(boxes_embedding, dim=-1)
    return boxes_embedding


def slice_and_pad(input_tensor, binary_mask):
    """
    Optimized version of slice and pad using PyTorch without explicit Python loops.
    
    input_tensor: Tensor of shape (B, C, H, W)
    binary_mask: Binary mask of shape (B, H, W)
    """
    B, C, H, W = input_tensor.shape

    # Get the indices where binary_mask is 1 for each sample in the batch
    flat_mask = binary_mask.view(B, -1)  # Flatten H, W for easier indexing
    num_elements = flat_mask.sum(dim=1)  # Number of valid elements (1s) per batch element
    max_len = num_elements.max().item()  # Maximum number of 1s across the batch

    # Create an index grid for efficient slicing
    batch_indices = th.arange(B).view(B, 1).expand(B, max_len).flatten()
    
    # Get the flattened indices of all valid positions (where binary_mask == 1)
    flattened_indices = [th.nonzero(flat_mask[b])[:max_len].squeeze(1) for b in range(B)]
    padded_indices = th.stack([F.pad(idx, (0, max_len - len(idx))) for idx in flattened_indices])
    
    # Reshape padded_indices to the original H, W layout
    h_indices = padded_indices // W
    w_indices = padded_indices % W

    # Gather the corresponding features based on the indices
    gathered_features = input_tensor[batch_indices, :, h_indices.flatten(), w_indices.flatten()]

    # Reshape the gathered features to (B, C, max_len)
    gathered_features = gathered_features.view(B, C, max_len)

    # Create the padding mask (False for valid positions, True for padded ones)
    padding_mask = th.arange(max_len).expand(B, max_len) >= num_elements.unsqueeze(1)

    return gathered_features, padding_mask