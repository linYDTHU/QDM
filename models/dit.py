# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

import numpy as np
import math
import warnings
from typing import Callable, List, Optional, Tuple, Union, Generator
"""
only for profiler.py
"""
# from timm.models.vision_transformer import Mlp

from timm.models.vision_transformer import Attention, Mlp



from timm.layers import use_fused_attn, Format, nchw_to, to_2tuple, _assert

if not use_fused_attn:
    warnings.warn("Fused kernel not available, using naive kernel")

def modulate(x, shift, scale):
    if len(x.shape) == 3:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif len(x.shape) == 4:
        return x * (1 + scale.unsqueeze(1).unsqueeze(2)) + shift.unsqueeze(1).unsqueeze(2)

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
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
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

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                         Modules for the lower stream                          #
#################################################################################
class CrossAttention(nn.Module):
    """
    Cross-attention block
    """
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        context_dim: int,  # context_dim is the dimensions of the context for the generation of the kv pairs
        num_heads: int = 8,
        q_bias: bool = False,
        kv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} has to be divided by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=q_bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=kv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.v_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_q(self, x: torch.Tensor) -> torch.Tensor:
        """
        compute query tensor.
        """
        B, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return self.q_norm(q)

    def _get_kv(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute key/value tensor
        """
        B, M, _ = context.shape
        # split kv via chunk，shape (B, M, num_heads, head_dim)
        k, v = self.kv(context).chunk(2, dim=-1)
        k = k.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return self.k_norm(k), self.v_norm(v)

    def _apply_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        according to fused_attn to get the output of attention.
        """
        if self.fused_attn:
            dropout_p = self.attn_drop.p if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            q = q * self.scale
            attn = torch.matmul(q, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            attn_out = torch.matmul(attn, v)
        return attn_out

    def forward(self, x: torch.Tensor, context: torch.Tensor, return_kv: bool = False) -> torch.Tensor:
        """
        forward pass
        """
        B, N, C = x.shape
        q = self._get_q(x)
        k, v = self._get_kv(context)

        attn_out = self._apply_attention(q, k, v)
        # restore shape and projection
        out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_kv:
            return out, k, v
        return out

    def forward_w_cache(self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        """
        use kv cache to compute forward pass.
        """
        B, N, C = x.shape
        q = self._get_q(x)
        attn_out = self._apply_attention(q, k_cache, v_cache)
        out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class GridPatchEmbed(nn.Module):
    def __init__(
        self, 
        patch_size: int, 
        large_patch_size: int, 
        img_size: int, 
        in_chans: int, 
        embed_dim: int, 
        norm_layer=nn.LayerNorm, 
        bias: bool = True
    ):
        """
        Initialize the module.

        Args:
            patch_size (int): The size of each small patch (assumed square).
            large_patch_size (int): The number of small patches contained in one large grid (assumed arranged in a square).
            img_size (int): The size of the image (assumed square).
            in_chans (int): The number of input channels.
            embed_dim (int): The embedding dimension.
            norm_layer (nn.Module): The normalization layer; default is LayerNorm.
            bias (bool): Whether to use bias in Conv2d.
        """
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.large_patch_size = to_2tuple(large_patch_size)
        self.embed_dim = embed_dim
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)
        
        # Ensure that grid_size is divisible by large_patch_size
        assert self.grid_size[0] % self.large_patch_size[0] == 0 and self.grid_size[1] % self.large_patch_size[1] == 0, \
            "Grid size must be divisible by large patch size."
        
        self.large_grid_size = (
            self.grid_size[0] // self.large_patch_size[0], 
            self.grid_size[1] // self.large_patch_size[1]
        )
        self.num_chunks = self.large_grid_size[0] * self.large_grid_size[1]
        # The shape of each large grid is determined by the small patch size and the number of patches in the large grid.
        self.large_grid_shape = tuple(s * p for s, p in zip(self.patch_size, self.large_patch_size))
        
        # Define the convolution layer for linear projection and the normalization layer.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.norm = norm_layer(embed_dim)
        
        # Define the positional embedding with shape 
        # (1, num_chunks, large_patch_size[0]*large_patch_size[1], embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_chunks, self.large_patch_size[0] * self.large_patch_size[1], embed_dim), 
            requires_grad=False
        )
        self.initialize_pos_embed()
    
    def _init_img_size(self, img_size: int) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
        """
        Initialize image size and grid information.

        Args:
            img_size (int): The size of the image.
        
        Returns:
            A tuple containing:
              - img_size as a tuple (H, W),
              - grid_size as a tuple (grid_h, grid_w),
              - the total number of patches.
        """
        img_size = to_2tuple(img_size)
        grid_size = tuple(s // p for s, p in zip(img_size, self.patch_size))
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches
    
    def _reshape_pos_embed(self, pos_embed: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Reshape the positional embedding.

        This converts a positional embedding of shape (1, grid_h, grid_w, embed_dim)
        into shape (1, num_chunks, large_patch_size[0]*large_patch_size[1], embed_dim).
        """
        # Permute to shape (1, embed_dim, grid_h, grid_w)
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        # Use unfold to partition the regions based on large_patch_size.
        pos_embed = F.unfold(pos_embed, kernel_size=self.large_patch_size, stride=self.large_patch_size)
        # Compute the new number of large grids.
        large_grid_size = (grid_size[0] // self.large_patch_size[0], grid_size[1] // self.large_patch_size[1])
        num_chunks = large_grid_size[0] * large_grid_size[1]
        pos_embed = pos_embed.reshape(1, self.embed_dim, self.large_patch_size[0], self.large_patch_size[1], num_chunks)
        pos_embed = pos_embed.permute(0, 4, 2, 3, 1).reshape(1, num_chunks, -1, self.embed_dim)
        return pos_embed

    def initialize_pos_embed(self) -> None:
        """
        Initialize (and freeze) the positional embedding using sine-cosine embedding.
        
        It is assumed that get_2d_sincos_pos_embed returns a NumPy array of shape (grid_h*grid_w, embed_dim).
        """
        pos_embed_np = get_2d_sincos_pos_embed(self.embed_dim, int(self.num_patches ** 0.5))
        pos_embed_tensor = torch.from_numpy(pos_embed_np).float().unsqueeze(0).reshape(1, self.grid_size[0], self.grid_size[1], -1)
        pos_embed_tensor = self._reshape_pos_embed(pos_embed_tensor, self.grid_size)
        self.pos_embed.data.copy_(pos_embed_tensor)
    
    def new_pos_embed(self, new_grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate a new positional embedding based on a new grid size (used when the input size changes).
        
        It is assumed that get_2d_sincos_pos_embed_new_grid returns a NumPy array based on the new grid size and the original grid size.
        """
        pos_embed_np = get_2d_sincos_pos_embed_new_grid(self.embed_dim, new_grid_size, self.grid_size)
        pos_embed_tensor = torch.from_numpy(pos_embed_np).float().unsqueeze(0).reshape(1, new_grid_size[0], new_grid_size[1], -1).to(self.pos_embed.device)
        return self._reshape_pos_embed(pos_embed_tensor, new_grid_size)
    
    def _get_pos_embed(self, image: torch.Tensor) -> torch.Tensor:
        """
        Get the appropriate positional embedding based on the input image size.
        If the image size differs from the initialized size, a new positional embedding is generated.
        """
        if image.shape[-2:] != self.img_size:
            grid_size = tuple(s // p for s, p in zip(image.shape[-2:], self.patch_size))
            return self.new_pos_embed(grid_size)
        else:
            return self.pos_embed
    
    def split_into_grids(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the input image x into large grid patches using unfold.
        
        Returns a tensor of shape (B, total_chunks, C, large_grid_h, large_grid_w).
        """
        B, C, H, W = x.shape
        # Unfold using the large_grid_shape (i.e., the size of one large grid).
        unfolded = F.unfold(x, kernel_size=self.large_grid_shape, stride=self.large_grid_shape)  # shape: (B, C * prod(large_grid_shape), num_chunks)
        num_chunks = unfolded.shape[-1]
        unfolded = unfolded.view(B, C, self.large_grid_shape[0], self.large_grid_shape[1], num_chunks)
        chunks = unfolded.permute(0, 4, 1, 2, 3)  # (B, num_chunks, C, large_grid_h, large_grid_w)
        return chunks

    def split_mask_into_grids(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Split the mask into grids according to the large grid size.
        
        The mask is expected to have its last two dimensions equal to grid_size.

        Returns a tensor of shape (B, num_chunks, prod(large_grid_shape)).
        """
        assert mask.shape[-2:] == self.grid_size, "Mask shape must be the same as the grid size."
        mask = F.unfold(mask.unsqueeze(1), kernel_size=self.large_patch_size, stride=self.large_patch_size)
        mask = mask.permute(0, 2, 1)
        return mask

    def forward(self, image: torch.Tensor, mask: torch.Tensor, num_chunks: Optional[int] = None) -> Generator[torch.Tensor, None, None]:
        """
        Forward pass (without using the mask for computation). Returns a generator yielding processed chunks.
        
        Args:
            image: Input image of shape (B, C, H, W).
            mask: Mask (not used in computation; provided for interface consistency).
            num_chunks: Number of chunks to process at a time. If None, all chunks are processed.
        """
        B = image.shape[0]
        pos_embed = self._get_pos_embed(image)

        if num_chunks is None:
            num_chunks = pos_embed.shape[1]
        else:
            assert num_chunks <= self.num_chunks, "Number of chunks must be less than the total number of chunks."
            assert num_chunks > 0, "Number of chunks must be greater than 0."
        
        chunks = self.split_into_grids(image)  # (B, total_chunks, C, h, w)
        # Process chunks in groups of 'num_chunks'
        for split_chunks, split_pos_emb in zip(chunks.split(num_chunks, dim=1), pos_embed.split(num_chunks, dim=1)):
            chunk_length = split_chunks.shape[1]
            # Merge batch and chunk dimensions to feed into the convolutional projection layer.
            split_chunks = split_chunks.reshape(-1, *chunks.shape[2:])  # (B*num_chunks, C, h, w)
            split_chunks = self.proj(split_chunks)  # (B*num_chunks, embed_dim, H_patches, W_patches)
            split_chunks = split_chunks.reshape(B, chunk_length, self.embed_dim, -1).permute(0, 1, 3, 2)  # (B, num_chunks, num_patches, embed_dim)
            
            # Normalize and add the corresponding positional embeddings.
            split_chunks = self.norm(split_chunks) + split_pos_emb
            yield split_chunks

    def forward_w_mask(self, image: torch.Tensor, mask: torch.Tensor, num_chunks: Optional[int] = 1) -> Generator[Tuple[int, int, torch.Tensor], None, None]:
        """
        Forward pass using a mask to limit computation (for inference only).
        Only processes chunks where the mask contains active regions (value 1) to save computation.
        
        Yields:
            A tuple of (batch_index, chunk_index, processed chunk).
        """
        B = image.shape[0]
        pos_embed = self._get_pos_embed(image) # (1, total_chunks, large_grid_h * large_grid_, C)
        
        if num_chunks is None:
            num_chunks = self.num_chunks
        else:
            assert num_chunks <= self.num_chunks, "Number of chunks must be less than the total number of chunks."
            assert num_chunks > 0, "Number of chunks must be greater than 0."
        
        chunks = self.split_into_grids(image)  # (B, total_chunks, C, large_grid_h, large_grid_w)
        mask_chunks = self.split_mask_into_grids(mask) # (B, total_chunks, large_grid_h*large_grid_w)

        total_chunks = chunks.shape[1]
        
        for batch_index in range(B):
            valid_chunks = []
            valid_chunk_indices = []

            # collect all valid chunks
            for chunk_index in range(total_chunks):
                if torch.any(mask_chunks[batch_index, chunk_index] == 1):  # valid chunk
                    chunk = chunks[batch_index, chunk_index].unsqueeze(0)  # (C, large_grid_h, large_grid_w)
                    pos_emb = pos_embed[0, chunk_index].unsqueeze(0)  # (1, large_grid_h * large_grid_w, C)

                    chunk = self.proj(chunk)  # project to embed_dim
                    chunk = chunk.reshape(1, self.embed_dim, -1).permute(0, 2, 1)  # (1, H*W, C)

                    # normalize and add positional embedding
                    chunk = self.norm(chunk) + pos_emb

                    valid_chunks.append(chunk)
                    valid_chunk_indices.append(chunk_index)

                # yied chunks if reach the num_chunks or the last chunk
                if len(valid_chunks) == num_chunks or chunk_index == total_chunks - 1:
                    if len(valid_chunks) != 0:
                        yield batch_index, torch.tensor(valid_chunk_indices, device=chunk.device), torch.stack(valid_chunks, dim=1)
                        valid_chunks = []
                        valid_chunk_indices = []

#################################################################################
#                                 Core DiT Model                                #
#################################################################################
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding with adaptable positional embeddings for any resolution."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
        bias: bool = True,
    ) -> None:
        """
        Args:
            img_size (int or tuple): Size of the input image.
            patch_size (int or tuple): Size of one patch.
            in_chans (int): Number of input channels.
            embed_dim (int): Dimension of the patch embedding.
            norm_layer (nn.Module, optional): Normalization layer to apply after adding positional embeddings.
            flatten (bool): If True, flatten the output to (B, N, embed_dim). If False, keep spatial dimensions.
            bias (bool): If True, adds a learnable bias to the convolution.
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Calculate grid size and number of patches.
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Projection layer to embed patches.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        
        # Positional embeddings (initialized for original grid size).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        
        # Normalization layer.
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

        self.initialize_pos_embed()

    def initialize_pos_embed(self) -> None:
        """Initialize (and freeze) positional embeddings using sine-cosine embeddings."""
        # Compute sine-cosine positional embeddings for a square grid.
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        # Copy the computed embeddings into the parameter.
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def new_pos_embed(self, new_grid_size: Tuple[int, int]) -> torch.Tensor:
        """
        Interpolate positional embeddings to match a new grid size.

        Args:
            new_grid_size (tuple): The new grid size (height, width).

        Returns:
            torch.Tensor: Interpolated positional embeddings with shape (1, new_num_patches, embed_dim).
        """
        pos_embed = get_2d_sincos_pos_embed_new_grid(self.pos_embed.shape[-1], new_grid_size, self.grid_size)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Embedded patches. The output shape is:
                - (B, N, embed_dim) if flatten is True, or
                - (B, embed_dim, H_patches, W_patches) if flatten is False.
        """
        B, C, H, W = x.shape

        # Validate that the image dimensions are divisible by the patch dimensions.
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            raise ValueError(f"Input image size ({H}x{W}) must be divisible by patch size {self.patch_size}.")

        # Compute new grid size based on input resolution.
        new_grid_size = (H // self.patch_size[0], W // self.patch_size[1])
        
        # Interpolate positional embeddings if the grid size has changed.
        if new_grid_size != self.grid_size:
            pos_embed = self.new_pos_embed(new_grid_size).to(x.device)
        else:
            pos_embed = self.pos_embed

        # Apply patch projection.
        x = self.proj(x)  # Shape: (B, embed_dim, H_patches, W_patches)

        if self.flatten:
            # Flatten spatial dimensions and transpose to (B, num_patches, embed_dim).
            x = x.flatten(2).transpose(1, 2)
            # Add positional embeddings.
            x = x + pos_embed
            # Apply normalization.
            x = self.norm(x)
        else:
            # Reshape pos_embed from (1, num_patches, embed_dim) to (1, embed_dim, H_patches, W_patches).
            pos_embed_reshaped = pos_embed.reshape(1, new_grid_size[0], new_grid_size[1], -1).permute(0, 3, 1, 2)
            # Add positional embeddings.
            x = x + pos_embed_reshaped
            # Apply normalization.
            x = self.norm(x)
            
        return x

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        cond_lq=True,
        lq_channels=3,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        base_channels = in_channels + lq_channels if cond_lq else in_channels
        self.cond_lq = cond_lq 

        self.x_embedder = PatchEmbed(input_size, patch_size, base_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
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

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

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
        h, w = self.x_embedder.grid_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, lq=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        lq: (N, C, H_lq, W_lq) tensor of low quality images (optional)
        """
        if lq is not None:
            assert self.cond_lq
            if lq.size(2) < x.size(2):
                lq = F.interpolate(lq, size=x.shape[-2:], mode="bicubic")
            x = torch.cat([x, lq], dim=1)

        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        c = t

        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

#################################################################################
#                                 Core QDM Model                                #
#################################################################################

class CrossAttnBlock(nn.Module):
    """
    A Cross-Attention block with adaptive layer norm zero (adaLN-Zero) conditioning modified from DiT block.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # Self-Attention Layers
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Cross-Attention Layers
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn2 = CrossAttention(hidden_size, hidden_size, num_heads=num_heads, q_bias=True, kv_bias=True, **block_kwargs)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp2 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, embeds, return_kv=False):
        B, chunk_size, N, D = x.shape

        # Self-Attention Forward Pass
        shift_msa, scale_msa, gate_msa, shift_mlp_1, scale_mlp_1, gate_mlp_1 = self.adaLN_modulation1(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1).unsqueeze(2) * self.attn1(modulate(self.norm1(x), shift_msa, scale_msa).reshape(B*chunk_size, N, D)).reshape(B, chunk_size, N, D)
        x = x + gate_mlp_1.unsqueeze(1).unsqueeze(2) * self.mlp1(modulate(self.norm2(x), shift_mlp_1, scale_mlp_1))

        # Cross-Attention Forward Pass
        x = x.reshape(B, chunk_size*N, D)
        shift_mca, scale_mca, gate_mca, shift_mlp_2, scale_mlp_2, gate_mlp_2 = self.adaLN_modulation2(c).chunk(6, dim=1)
        residual = x
        x = modulate(self.norm3(x), shift_mca, scale_mca)
        if return_kv:
            x, k, v = self.attn2(x, embeds, return_kv=return_kv)
        else:
            x = self.attn2(x, embeds, return_kv=return_kv)
        x = residual + gate_mca.unsqueeze(1) * x
        x = x + gate_mlp_2.unsqueeze(1) * self.mlp2(modulate(self.norm4(x), shift_mlp_2, scale_mlp_2))

        x = x.reshape(B, chunk_size, N, D)

        if return_kv:
            return x, k, v
        else:
            return x
    
    def forward_w_cache(self, x, c, k_cache, v_cache):
        B, chunk_size, N, D = x.shape
        x = x.reshape(B * chunk_size, N, D)
        
        # Self-Attention Forward Pass
        shift_msa, scale_msa, gate_msa, shift_mlp_1, scale_mlp_1, gate_mlp_1 = self.adaLN_modulation1(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1).unsqueeze(2) * self.attn1(modulate(self.norm1(x), shift_msa, scale_msa).reshape(B*chunk_size, N, D)).reshape(B, chunk_size, N, D)
        x = x + gate_mlp_1.unsqueeze(1).unsqueeze(2) * self.mlp1(modulate(self.norm2(x), shift_mlp_1, scale_mlp_1))

        # Cross-Attention Forward Pass
        x = x.reshape(B, chunk_size*N, D)
        shift_mca, scale_mca, gate_mca, shift_mlp_2, scale_mlp_2, gate_mlp_2 = self.adaLN_modulation2(c).chunk(6, dim=1)
        residual = x
        x = modulate(self.norm3(x), shift_mca, scale_mca)
        x = self.attn2.forward_w_cache(x, k_cache, v_cache)
        x = residual + gate_mca.unsqueeze(1) * x
        x = x + gate_mlp_2.unsqueeze(1) * self.mlp2(modulate(self.norm4(x), shift_mlp_2, scale_mlp_2))

        x = x.reshape(B, chunk_size, N, D)

        return x

class UpStream(nn.Module):
    """
    The upper stream with a series of DiT blocks
    """
    def __init__(
        self,
        input_size=32,
        patch_size=8,
        in_channels=4,
        out_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels =  out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

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

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, grid_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if grid_size is None:
            h, w = self.x_embedder.grid_size
        else:
            h, w = grid_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, return_preds=False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        grid_size = x.shape[-2:]
        grid_size = (grid_size[0] // self.patch_size, grid_size[1] // self.patch_size)
        x = self.x_embedder(x)                   # (N, T, D), where T = H * W / patch_size ** 2
        c = self.t_embedder(t)                   # (N, D)

        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        if return_preds:
            pred = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
            pred = self.unpatchify(pred, grid_size)                   # (N, out_channels, H, W)
            return x, pred
        else:
            return x

class DownStream(nn.Module):
    """
    The lower stream with cross-attention blocks that each forward deals with a chunk of the input.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        large_patch_size=4,
        in_channels=4,
        out_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels =  out_channels
        self.patch_size = patch_size
        self.large_patch_size = large_patch_size
        self.num_heads = num_heads

        self.x_embedder = GridPatchEmbed(patch_size, large_patch_size, input_size, in_channels, hidden_size, bias=True)

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            CrossAttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
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

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation1[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation1[-1].bias, 0)
            nn.init.constant_(block.adaLN_modulation2[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation2[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x, t, embeds, mask, up_pred=None, chunk_size:int=None):
        """
        Forward pass of Downstream. This forward pass doesn't use mask to save computation.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        embeds: A list of (N, D) tensor of embeddings from upper stream
        mask: (N, num_patches) tensor of masks where 1 indicates an active site
        """
        N, C, H, W = x.shape
        grid_size = (H // self.patch_size, W // self.patch_size)
        large_grid_size  = (grid_size[0]//self.large_patch_size, grid_size[1]//self.large_patch_size)
        num_chunks = large_grid_size[0] * large_grid_size[1]
        large_grid_shape = (self.large_patch_size * self.patch_size, self.large_patch_size * self.patch_size)
        # up_pred = x if up_pred is None else up_pred
        # Context computation
        c = self.t_embedder(t)  # (N, D)

        x_all = []
        for index, x in enumerate(self.x_embedder(x, mask, chunk_size)):  # (N, chunk_size, H_patches*W_patches, D)
            if index == 0:
                k_all = []
                v_all = []
                for i, block in enumerate(self.blocks):
                    x, k, v = block(x, c, embeds, return_kv=True) # (N, chunk_size, H_patches*W_patches, D)
                    k_all.append(k)
                    v_all.append(v)
                x = self.final_layer(x, c)  # (N, chunk_size, H_patches*W_patches, patch_size ** 2 * out_channels)
            else:
                for i, block in enumerate(self.blocks):
                    x = block.forward_w_cache(x, c, k_all[i], v_all[i]) # (N, chunk_size, H_patches*W_patches, D)
                    # x = block(x, c, embeds, return_kv=False)
                x = self.final_layer(x, c)  # (N, chunk_size, H_patches*W_patches, patch_size ** 2 * out_channels)
            x_all.append(x)
        x = torch.cat(x_all, dim=1)  # (N, num_chunks, H_patches*W_patches, patch_size ** 2 * out_channels)
        x = x.reshape(N, num_chunks, self.large_patch_size, self.large_patch_size, self.patch_size, self.patch_size, self.out_channels).permute(0,1,2,4,3,5,6)
        x = x.reshape(N, large_grid_size[0], large_grid_size[1], large_grid_shape[0], large_grid_shape[1], self.out_channels).permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(N, self.out_channels, H, W)
        return x
    
    def forward_w_mask(self, x, t, embeds, mask, up_pred, chunk_size:int=1):
        """
        Forward pass of Downstream. Consider input mask to save computation.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        embeds: A list of (N, T, D) tensor of embeddings from upper stream
        mask: (N, grid_size, grid_size) tensor of masks where 1 indicates an active site
        up_pred: (N, C, H, W) tensor of predictions from the upper stream
        """
        N, C, H, W = x.shape
        grid_size = (H // self.patch_size, W // self.patch_size)
        large_grid_size  = (grid_size[0]//self.large_patch_size, grid_size[1]//self.large_patch_size)
        num_chunks = large_grid_size[0] * large_grid_size[1]
        large_grid_shape = (self.large_patch_size * self.patch_size, self.large_patch_size * self.patch_size)
        # Context computation
        c = self.t_embedder(t)  # (N, D)

        if chunk_size is None:
            chunk_size = num_chunks

        x_all = torch.zeros(N, num_chunks, self.large_patch_size*self.large_patch_size, self.patch_size*self.patch_size*self.out_channels, device=x.device, dtype=x.dtype)
        current_batch_index = -1
        for index, data in enumerate(self.x_embedder.forward_w_mask(x, mask, chunk_size)):  # (N, chunk_size, H_patches*W_patches, D)
            batch_index, chunk_index, x = data
            if batch_index != current_batch_index:
                k_all = []
                v_all = []
                current_batch_index = batch_index
                for i, block in enumerate(self.blocks):
                    x, k, v = block(x, c[[batch_index]], embeds[[batch_index]], return_kv=True) # (1, chunk_size, H_patches*W_patches, D)
                    k_all.append(k)
                    v_all.append(v)
                x = self.final_layer(x, c[[batch_index]])  # (1, chunk_size, H_patches*W_patches, patch_size ** 2 * out_channels)
            else:
                for i, block in enumerate(self.blocks):
                    x = block.forward_w_cache(x, c[[batch_index]], k_all[i], v_all[i]) # (1, chunk_size, H_patches*W_patches, D)
                    # x = block(x, c, embeds, return_kv=False)
                x = self.final_layer(x, c[[batch_index]])  # (1, chunk_size, H_patches*W_patches, patch_size ** 2 * out_channels)
            x_all[batch_index, chunk_index] = x[0].to(dtype=x_all.dtype)
        x = x_all
        _, num_chunks, _, _ = x.shape
        x = x.reshape(N, num_chunks, self.large_patch_size, self.large_patch_size, self.patch_size, self.patch_size, self.out_channels).permute(0,1,2,4,3,5,6)
        x = x.reshape(N, large_grid_size[0], large_grid_size[1], large_grid_shape[0], large_grid_shape[1], self.out_channels).permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(N, self.out_channels, H, W)

        # upsample mask to the same size as the output
        mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest')
        # combine the output from two streams
        x = x * mask + up_pred * (1 - mask)

        return x

class QDM(nn.Module):
    """
    Scalable Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        up_patch_size=8,
        down_patch_size=2,
        down_large_patch_size=4,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        cond_lq=True,
        lq_channels=3,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.up_patch_size = up_patch_size
        self.down_patch_size = down_patch_size
        self.num_heads = num_heads

        base_channels = in_channels + lq_channels if cond_lq else in_channels
        self.cond_lq = cond_lq 

        self.upstream = UpStream(input_size, up_patch_size, base_channels, in_channels, hidden_size, depth, num_heads, mlp_ratio)
        self.downstream = DownStream(input_size, down_patch_size, down_large_patch_size, base_channels, in_channels, hidden_size, depth, num_heads, mlp_ratio)

    def forward(self, x, t, mask, lq=None, chunk_size:int=None, up_pred=False):
        """
        Forward pass of DiT. This forward process doesn't use mask to save computation.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        mask: (N, grid_size, grid_size) tensor of masks where 1 indicates an active site
        chunk_size: Size of each chunk to process.
        """
        if lq is not None:
            assert self.cond_lq
            if lq.size(2) < x.size(2):
                lq = F.interpolate(lq, size=x.shape[-2:], mode="bicubic")
            x = torch.cat([x, lq], dim=1)

        embeds, pred = self.upstream(x, t, return_preds=True)
        x = self.downstream(x, t, embeds, mask, pred, chunk_size)

        if up_pred:
            return x, pred
        else:
            return x
        
    def forward_w_mask(self, x, t, mask, lq=None, chunk_size:int=None, up_pred=False):
        """
        Forward pass of DiT. This forward process uses mask to save computation.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        mask: (N, grid_size, grid_size) tensor of masks where 1 indicates an active site
        chunk_size: Size of each chunk to process.
        """
        if lq is not None:
            assert self.cond_lq
            if lq.size(2) < x.size(2):
                lq = F.interpolate(lq, size=x.shape[-2:], mode="bicubic")
            x = torch.cat([x, lq], dim=1)

        embeds, pred = self.upstream(x, t, return_preds=True)
        x = self.downstream.forward_w_mask(x, t, embeds, mask, pred, chunk_size)

        if up_pred:
            return x, pred
        else:
            return x

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

def get_2d_sincos_pos_embed_new_grid(embed_dim: int,
                                      grid_size: Tuple[int, int],
                                      old_grid_size: Tuple[int, int],
                                      cls_token: bool = False,
                                      extra_tokens: int = 0) -> np.ndarray:
    """
    Generate 2D sine-cosine positional embeddings for a new grid by interpolating
    from the original grid size.

    Args:
        embed_dim (int): Dimension of the embedding.
        grid_size (Tuple[int, int]): The new grid size as (height, width).
        old_grid_size (Tuple[int, int]): The original grid size as (height, width)
                                         used for the initial embedding.
        cls_token (bool, optional): If True, reserve additional tokens for class tokens.
        extra_tokens (int, optional): Number of extra tokens (e.g., class tokens) to be prepended.

    Returns:
        np.ndarray: Positional embeddings of shape [grid_size[0]*grid_size[1], embed_dim] or
                    [extra_tokens + grid_size[0]*grid_size[1], embed_dim] if cls_token is True.
    """
    # Validate that grid_size and old_grid_size are tuples of two integers.
    if len(grid_size) != 2 or len(old_grid_size) != 2:
        raise ValueError("grid_size and old_grid_size must be tuples of two integers.")

    # Scale new grid coordinates to the original grid's scale.
    # This ensures that the new grid is aligned with the original grid embedding space.
    grid_h = (np.arange(grid_size[0], dtype=np.float32) / grid_size[0]) * old_grid_size[0]
    grid_w = (np.arange(grid_size[1], dtype=np.float32) / grid_size[1]) * old_grid_size[1]
    
    # Create a meshgrid.
    # Note: grid_w is used for the x-axis and grid_h for the y-axis.
    mesh_w, mesh_h = np.meshgrid(grid_w, grid_h)  # mesh_w and mesh_h shapes: (grid_size[0], grid_size[1])
    
    # Stack the meshgrid to create a 2-channel grid.
    grid = np.stack([mesh_w, mesh_h], axis=0)  # shape: (2, grid_size[0], grid_size[1])
    
    # Expand dimensions to match the expected shape for get_2d_sincos_pos_embed_from_grid.
    grid = np.expand_dims(grid, axis=1)  # shape: (2, 1, grid_size[0], grid_size[1])
    
    # Generate sine-cosine embeddings from the grid.
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    # If class tokens are used, prepend extra token embeddings (e.g., zeros) to the positional embeddings.
    if cls_token and extra_tokens > 0:
        extra_embed = np.zeros((extra_tokens, embed_dim), dtype=pos_embed.dtype)
        pos_embed = np.concatenate([extra_embed, pos_embed], axis=0)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT/QDM Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def QDM_S_u8_l2(**kwargs):
    return QDM(depth=6, hidden_size=384, up_patch_size=8, down_patch_size=2, num_heads=6, **kwargs)

def QDM_B_u8_l2(**kwargs):
    return QDM(depth=6, hidden_size=768, up_patch_size=8, down_patch_size=2, num_heads=12, **kwargs)

def QDM_L_u8_l2(**kwargs):
    return QDM(depth=12, hidden_size=1024, up_patch_size=8, down_patch_size=2, num_heads=16, **kwargs)

def QDM_XL_u8_l2(**kwargs):
    return QDM(depth=14, hidden_size=1152, up_patch_size=8, down_patch_size=2, num_heads=16, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}

QDM_models = {
    'QDM-S-u8-l2': QDM_S_u8_l2,
    'QDM-B-u8-l2': QDM_B_u8_l2,
    'QDM-L-u8-l2': QDM_L_u8_l2,
    'QDM-XL-u8-l2': QDM_XL_u8_l2,
}

#################################################################################
#                               Some test code                                  #
#################################################################################

def test_qdm_model_on_gpu():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Define the input parameters
    batch_size = 1
    input_size = 1024  # spatial dimensions (height and width)
    down_large_patch_size = 8
    in_channels = 3
    mlp_ratio = 4.0
    learn_sigma = False
    cond_lq = True
    lq_channels = 3

    # Initialize the model and move it to the GPU
    model = QDM_L_u8_l2(
        input_size=input_size,
        down_large_patch_size = down_large_patch_size,
        in_channels=in_channels,
        mlp_ratio=mlp_ratio,
        learn_sigma=learn_sigma,
        cond_lq=cond_lq,
        lq_channels=lq_channels
    ).to(device)
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total number of parameters in QDM: {total_params/1000**2:.2f}M")

    # Reset peak memory stats and print initial memory status
    torch.cuda.reset_peak_memory_stats(device)
    initial_allocation = torch.cuda.memory_allocated(device)
    initial_reserved = torch.cuda.memory_reserved(device) 

    # Define dummy inputs and move them to GPU
    x = torch.randn(batch_size, in_channels, input_size, input_size, device=device)  # Input tensor
    lq = torch.randn(batch_size, 3, input_size, input_size, device=device)  # Latent query tensor
    t = torch.randint(0, 15, (batch_size,), device=device)  # Diffusion timesteps
    # y = torch.randint(0, num_classes, (batch_size,), device=device)  # Class labels
    mask = torch.ones(batch_size, model.downstream.x_embedder.num_patches, device=device)  # Mask tensor

    # Forward pass (without classifier-free guidance)
    with torch.no_grad():
        output = model(x, t, mask, lq=lq, chunk_size=64)
        # output_ = model(x, t, y, mask, chunk_size=model.downstream.x_embedder.num_patches//10)
        # print((output-output_).abs().sum())

    # Print peak GPU memory utilization after forward pass
    print(f"Peak memory allocated during forward pass: {(torch.cuda.max_memory_allocated(device)-initial_allocation) / 1024 ** 3:.2f} GB")
    print(f"Peak memory reserved during forward pass: {(torch.cuda.max_memory_reserved(device)-initial_reserved) / 1024 ** 3:.2f} GB")

    # Expected output shape check
    out_channels = in_channels * 2 if learn_sigma else in_channels
    expected_shape = (batch_size, out_channels, input_size, input_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Forward pass output shape test passed.")

    num_warmup_runs = 5  # Number of warm-up runs to stabilize the model
    num_measurement_runs = 5  # Number of runs to average over for accurate timing
    # Warm-up on GPU
    with torch.no_grad():
        for _ in range(num_warmup_runs):
            _ = model(x, t, mask, lq=lq, chunk_size=64)
    print("Warm-up completed.")

    # Measure time for multiple runs on GPU
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    gpu_times = []

    with torch.no_grad():
        for _ in range(num_measurement_runs):
            start_event.record()
            _ = model(x, t, mask, lq=lq, chunk_size=64)
            end_event.record()
            torch.cuda.synchronize()  # Ensure that all operations are completed
            gpu_times.append(start_event.elapsed_time(end_event))  # Time in milliseconds

    average_gpu_time = sum(gpu_times) / num_measurement_runs
    print(f"Average Inference Time on GPU: {average_gpu_time:.3f} ms")

def test_dit_model_on_gpu():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Define the input parameters
    batch_size = 1
    input_size = 128  # spatial dimensions (height and width)
    patch_size = 8
    in_channels = 3
    hidden_size = 384
    depth = 12
    num_heads = 6
    mlp_ratio = 4.0
    num_classes = 1000
    learn_sigma = True

    # Initialize the model and move it to the GPU
    model = DiT(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        class_dropout_prob=0.1,
        num_classes=num_classes,
        learn_sigma=learn_sigma
    ).to(device)
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total number of parameters in DiT: {total_params/1000**2:.2f}M")

    # Reset peak memory stats and print initial memory status
    torch.cuda.reset_peak_memory_stats(device)
    initial_allocation = torch.cuda.memory_allocated(device)
    initial_reserved = torch.cuda.memory_reserved(device) 

    # Define dummy inputs and move them to GPU
    x = torch.randn(batch_size, in_channels, input_size, input_size, device=device)  # Input tensor
    t = torch.randint(0, 1000, (batch_size,), device=device)  # Diffusion timesteps
    y = torch.randint(0, num_classes, (batch_size,), device=device)  # Class labels

    # Forward pass (without classifier-free guidance)
    with torch.no_grad():
        output = model(x, t, y)

    # Print peak GPU memory utilization after forward pass
    print(f"Peak memory allocated during forward pass: {(torch.cuda.max_memory_allocated(device)-initial_allocation) / 1024 ** 3:.2f} GB")
    print(f"Peak memory reserved during forward pass: {(torch.cuda.max_memory_reserved(device)-initial_reserved) / 1024 ** 3:.2f} GB")

    # Expected output shape check
    out_channels = in_channels * 2 if learn_sigma else in_channels
    expected_shape = (batch_size, out_channels, input_size, input_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Forward pass output shape test passed.")

    num_warmup_runs = 0  # Number of warm-up runs to stabilize the model
    num_measurement_runs = 1  # Number of runs to average over for accurate timing
    # Warm-up on GPU
    with torch.no_grad():
        for _ in range(num_warmup_runs):
            _ = model(x, t, y)
    print("Warm-up completed.")

    # Measure time for multiple runs on GPU
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    gpu_times = []

    with torch.no_grad():
        for _ in range(num_measurement_runs):
            start_event.record()
            _ = model(x, t, y)
            end_event.record()
            torch.cuda.synchronize()  # Ensure that all operations are completed
            gpu_times.append(start_event.elapsed_time(end_event))  # Time in milliseconds

    average_gpu_time = sum(gpu_times) / num_measurement_runs
    print(f"Average Inference Time on GPU: {average_gpu_time:.3f} ms")

    # Forward pass
    output = model(x, t, y)

    # Define a dummy loss function (e.g., MSE)
    target = torch.randn_like(output)  # Dummy target tensor
    loss_fn = torch.nn.MSELoss()  # Mean Squared Error Loss
    loss = loss_fn(output, target)  # Compute loss

    # Backward pass
    loss.backward()  # Compute gradients
    print("Backward pass completed successfully.")

    # Check if gradients exist for model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or param.grad.abs().sum() == 0:
                print(f"Parameter '{name}' has no gradient!")
    print("All gradients are properly computed.")

# Run the test function
if __name__ == "__main__":
    test_qdm_model_on_gpu()
    # test_dit_model_on_gpu()