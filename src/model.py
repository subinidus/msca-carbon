"""
================================================================================
model.py -- Multi-Stream Cross-Attention (MS-CA) Architecture (ViT backbone)
================================================================================

Purpose
-------
Model architecture for the MS-CA satellite-data fusion network, refactored for a
STRICTLY SINGLE-TASK, PURE-REGRESSION objective (continuous carbon-emission
value). All classification logic has been removed.

    Stream A (Environment Proxy)          : [Batch, 3, 16, 16]
        Channels: NO2, SO2, CO
    Stream B (Socio-Infrastructure Proxy) : [Batch, 4, 16, 16]
        Channels: Nightlight, Urban Fraction, Power Plant, Fossil Capacity

Note on Stream B: population data has been excluded, so Stream B is now 4
channels (was 5). This is reflected in ``in_channels_b=4`` on ``MSCANet``.

What changed vs. the multi-task version
----------------------------------------
1. NO CLASSIFICATION. The 5-class hotspot ``ClassificationHead`` has been
   deleted. ``MSCANet`` now owns a single ``RegressionHead`` and its
   ``forward`` returns ONLY the scalar regression output.
2. FIXED CHANNELS. Stream B is 4 channels (population removed); Stream A stays
   at 3. Channel counts are surfaced as ``in_channels_a`` / ``in_channels_b``
   constructor arguments.

Backbone: Vision Transformer (ViT)
----------------------------------
A standard ``ViTPatchEncoder`` (Conv2d patch-embed -> learned positional
embeddings -> Transformer encoder blocks) is used for both streams. Inputs are
FIXED at 16x16 spatial size with a patch size of 4x4, giving a 4x4 = 16 token
grid per stream (Sequence_Length = 16).

Design goals (unchanged)
------------------------
1. STRICT MODULARITY: Encoders are built behind the abstract ``BaseEncoder``
   interface. ``ViTPatchEncoder`` is a drop-in implementation -- to swap in a
   different backbone later, subclass ``BaseEncoder`` and pass the new class
   into ``MSCANet(encoder_a_cls=..., encoder_b_cls=...)``. Nothing in the fusion
   layer or the regression head needs to change.
2. EXPLAINABILITY: ``CrossAttentionFusion`` still returns the raw per-head
   attention score matrix for XAI, reshapeable via
   ``reshape_attention_to_spatial_grid``.

This file contains NO dataset, training loop, or inference demo code -- it is
pure architecture, imported by the training/evaluation scripts
(``from model import MSCANet``).
================================================================================
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


# ==============================================================================
# SECTION 1: MODULAR ENCODER INTERFACE
# ==============================================================================
class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class that every stream-encoder backbone must implement.

    Contract
    --------
    forward(x) : x is a raw image tensor [Batch, Channels, H, W] and MUST
                 return a token sequence [Batch, Sequence_Length, Embed_Dim].
    num_patches: the fixed sequence length produced by the encoder.
    grid_size  : (grid_h, grid_w) so the token sequence can be reshaped back
                 into a 2D spatial map -- this is what makes the attention
                 matrices interpretable/visualizable later.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_patches(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def grid_size(self) -> tuple[int, int]:
        raise NotImplementedError


class TransformerEncoderBlock(nn.Module):
    """
    A single pre-norm Transformer encoder block (self-attention + MLP).
    Used internally by the ViT backbone.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Self-attention sub-block (pre-norm + residual) ---
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + attn_out

        # --- MLP sub-block (pre-norm + residual) ---
        x = x + self.mlp(self.norm2(x))
        return x


# ==============================================================================
# SECTION 2: VISION TRANSFORMER (ViT) PATCH ENCODER
# ==============================================================================
class ViTPatchEncoder(BaseEncoder):
    """
    Lightweight Vision Transformer encoder.

    Pipeline: Conv2d patch-embedding -> add learned positional embeddings
    -> N Transformer encoder blocks -> final LayerNorm.

    Fixed for this pipeline's 16x16 input crops with a patch size of 4x4:
    grid_h = grid_w = 16 / 4 = 4, so num_patches = 4 * 4 = 16 tokens.

    No [CLS] token is used on purpose: keeping the output sequence purely
    spatial (one token per patch) means the sequence length always equals
    grid_h * grid_w, which is required to cleanly reshape attention maps back
    into a 2D grid for XAI visualization.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: tuple[int, int] = (16, 16),
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        **_unused_kwargs,
    ):
        """
        Args:
            in_channels : number of input channels for this stream
                          (3 for Stream A, 4 for Stream B).
            img_size    : (H, W) of the raw input raster. Defaults to the
                          pipeline's fixed 16x16 crop size.
            patch_size  : side length of each square patch. Defaults to 4,
                          giving a 4x4 = 16 token grid for a 16x16 input.
            embed_dim   : token embedding dimension.
            depth       : number of stacked Transformer encoder blocks.
            num_heads   : number of self-attention heads per block.
            dropout     : dropout probability used throughout the block.
            **_unused_kwargs: absorbs any extra keyword arguments so
                          ``MSCANet`` can stay fully backbone-agnostic without
                          every encoder sharing an identical constructor
                          signature.
        """
        super().__init__(embed_dim)
        img_h, img_w = img_size
        assert img_h % patch_size == 0 and img_w % patch_size == 0, (
            "Image dimensions must be divisible by patch_size."
        )

        self.patch_size = patch_size
        self.grid_h = img_h // patch_size
        self.grid_w = img_w // patch_size
        self._num_patches = self.grid_h * self.grid_w

        # Patch embedding: a single strided conv turns each non-overlapping
        # patch_size x patch_size patch into one embed_dim-length vector.
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Learned positional embeddings, one per patch/token.
        self.pos_embed = nn.Parameter(torch.zeros(1, self._num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def grid_size(self) -> tuple[int, int]:
        return (self.grid_h, self.grid_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]  (e.g. [B, 3, 16, 16] or [B, 4, 16, 16])
        x = self.patch_embed(x)                 # [B, embed_dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)        # [B, N, embed_dim] -> [B, 16, D]
        x = x + self.pos_embed                   # broadcast add positional info
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x                                # [B, N, embed_dim]


# ==============================================================================
# SECTION 3: CROSS-ATTENTION FUSION LAYER
# ==============================================================================
class CrossAttentionFusion(nn.Module):
    """
    Fuses the two encoded streams via cross-attention.

        Query (Q)       <- Stream B tokens (Socio-Infrastructure)
        Key/Value (K,V) <- Stream A tokens (Environment)

    This design asks: "given each socio-infrastructure patch, which
    environmental (NO2/SO2/CO) patches are most relevant?" -- intuitive for
    emission attribution.

    The raw attention weight matrix is returned unmodified (per attention head)
    for downstream Explainable-AI visualization.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(
        self, tokens_b: torch.Tensor, tokens_a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens_b: [Batch, Lq, Embed_Dim]  (Query source: Stream B)
            tokens_a: [Batch, Lk, Embed_Dim]  (Key/Value source: Stream A)

        Returns:
            fused_tokens : [Batch, Lq, Embed_Dim]
            attn_weights : [Batch, Num_Heads, Lq, Lk]  (per-head, for XAI)
        """
        q = self.norm_q(tokens_b)
        kv = self.norm_kv(tokens_a)

        fused, attn_weights = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=True,
            average_attn_weights=False,
        )

        # Residual connection back onto the query stream (Stream B), so the
        # fused output still carries the socio-infra identity.
        fused_tokens = self.norm_out(fused + tokens_b)
        return fused_tokens, attn_weights


# ==============================================================================
# SECTION 4: REGRESSION OUTPUT HEAD
# ==============================================================================
class RegressionHead(nn.Module):
    """Scalar carbon-emission value regression head (the ONLY task head)."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # [Batch, 1]


# ==============================================================================
# SECTION 5: FULL MS-CA MODEL (single-task regression)
# ==============================================================================
class MSCANet(nn.Module):
    """
    Full Multi-Stream Cross-Attention network for single-task regression.

    Composition (all injected as sub-modules, nothing hard-wired):
        encoder_a  : BaseEncoder subclass for Stream A (Environment, 3ch)
        encoder_b  : BaseEncoder subclass for Stream B (Socio-Infra, 4ch)
        fusion     : CrossAttentionFusion
        reg_head   : RegressionHead  (single scalar output)

    To swap the backbone later, only the ``encoder_a_cls`` / ``encoder_b_cls``
    constructor arguments need to change.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (16, 16),
        patch_size: int = 4,
        embed_dim: int = 128,
        encoder_depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        in_channels_a: int = 3,
        in_channels_b: int = 4,
        encoder_a_cls=ViTPatchEncoder,
        encoder_b_cls=ViTPatchEncoder,
        **encoder_kwargs,
    ):
        """
        Args:
            img_size: (H, W) of the raw input crop. Defaults to the pipeline's
                fixed 16x16 size.
            patch_size: side length of each square patch (default 4 ->
                Sequence_Length = 16).
            embed_dim, encoder_depth, dropout: shared hyperparameters forwarded
                to both stream encoders.
            num_heads: number of attention heads, used by BOTH the ViT encoders'
                self-attention AND the cross-attention fusion layer.
            in_channels_a: Stream A channel count (NO2, SO2, CO) -> 3.
            in_channels_b: Stream B channel count (Nightlight, Urban Fraction,
                Power Plant, Fossil Capacity) -> 4. Population has been excluded.
            encoder_a_cls / encoder_b_cls: ``BaseEncoder`` subclasses used to
                build Stream A's and Stream B's encoders -- the single point of
                control for backbone swaps.
            **encoder_kwargs: extra backbone-specific hyperparameters forwarded
                verbatim to both encoder constructors.
        """
        super().__init__()
        self.embed_dim = embed_dim

        common_encoder_kwargs = dict(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            dropout=dropout,
            **encoder_kwargs,
        )

        # Independent encoders -> weights are NOT shared between Stream A and
        # Stream B, since the two modalities have different statistics.
        #   Stream A: 3 channels (NO2, SO2, CO)
        #   Stream B: 4 channels (Nightlight, Urban Fraction, Power Plant,
        #             Fossil Capacity)  [population excluded]
        self.encoder_a = encoder_a_cls(in_channels=in_channels_a, **common_encoder_kwargs)
        self.encoder_b = encoder_b_cls(in_channels=in_channels_b, **common_encoder_kwargs)

        self.fusion = CrossAttentionFusion(embed_dim, num_heads=num_heads, dropout=dropout)

        # Single-task: the ONLY head is the scalar regression head.
        self.reg_head = RegressionHead(embed_dim, dropout=dropout)

    def forward(
        self,
        stream_a: torch.Tensor,
        stream_b: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            stream_a: [Batch, 3, 16, 16]  Environment proxy (NO2, SO2, CO)
            stream_b: [Batch, 4, 16, 16]  Socio-infrastructure proxy
            return_attention: if True, also return the raw XAI attention
                              weight matrix from the fusion layer.

        Returns:
            reg_out : [Batch, 1]  (continuous carbon-emission prediction)
            (optional) attn_weights : [Batch, Num_Heads, Lq, Lk]
        """
        tokens_a = self.encoder_a(stream_a)   # [B, Na, D] -> [B, 16, D]
        tokens_b = self.encoder_b(stream_b)   # [B, Nb, D] -> [B, 16, D]

        fused_tokens, attn_weights = self.fusion(tokens_b, tokens_a)  # [B, Nb, D]

        # Global average pooling over the sequence dimension -> one feature
        # vector per sample before the regression head.
        pooled = fused_tokens.mean(dim=1)     # [B, D]

        reg_out = self.reg_head(pooled)       # [B, 1]

        if return_attention:
            return reg_out, attn_weights
        return reg_out


# ==============================================================================
# SECTION 6: XAI HELPER -- RESHAPE ATTENTION BACK TO A 2D SPATIAL MAP
# ==============================================================================
def reshape_attention_to_spatial_grid(
    attn_weights: torch.Tensor,
    query_grid: tuple[int, int],
    key_grid: tuple[int, int],
    average_heads: bool = True,
) -> torch.Tensor:
    """
    Converts the flat cross-attention matrix into a spatial tensor suitable for
    visual mapping (e.g. overlaying on the input raster).

    Args:
        attn_weights : [Batch, Num_Heads, Lq, Lk] as returned by
                       CrossAttentionFusion.
        query_grid   : (grid_h, grid_w) of the Query source encoder
                       (Stream B / socio-infra), with grid_h*grid_w == Lq.
        key_grid     : (grid_h, grid_w) of the Key/Value source encoder
                       (Stream A / environment), with grid_h*grid_w == Lk.
        average_heads: if True, average across attention heads first.

    Returns:
        If average_heads:
            [Batch, gh_q, gw_q, gh_k, gw_k]
        else:
            [Batch, Num_Heads, gh_q, gw_q, gh_k, gw_k]
    """
    gh_q, gw_q = query_grid
    gh_k, gw_k = key_grid

    if average_heads:
        attn = attn_weights.mean(dim=1)  # [B, Lq, Lk]
        B, Lq, Lk = attn.shape
        assert Lq == gh_q * gw_q, "Query grid does not match sequence length."
        assert Lk == gh_k * gw_k, "Key grid does not match sequence length."
        return attn.view(B, gh_q, gw_q, gh_k, gw_k)
    else:
        B, num_heads, Lq, Lk = attn_weights.shape
        assert Lq == gh_q * gw_q, "Query grid does not match sequence length."
        assert Lk == gh_k * gw_k, "Key grid does not match sequence length."
        return attn_weights.view(B, num_heads, gh_q, gw_q, gh_k, gw_k)
