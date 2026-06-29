from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.utils import zero_module
from modules.transformer import AbsolutePositionEmbedder
from modules.norm import LayerNorm32
from modules import sparse as sp
from modules.sparse.transformer import ModulatedSparseTransformerPureSelfBlock

class SparseResBlock3d(nn.Module):
    def __init__(self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels, bias=True))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        x = self._updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    

class SLatFlowModel(nn.Module):
    def __init__(self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm

        if self.io_block_channels is not None:
            assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
            assert np.log2(patch_size) == len(io_block_channels), "Number of IO ResBlocks must match the number of stages"

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels if io_block_channels is None else io_block_channels[0])
        
        self.input_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                self.input_blocks.extend([SparseResBlock3d(chs, model_channels, out_channels=chs) for _ in range(num_io_res_blocks-1)])
                self.input_blocks.append(SparseResBlock3d(chs, model_channels, out_channels=next_chs, downsample=True))
            
        self.blocks = nn.ModuleList([ModulatedSparseTransformerPureSelfBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm) for _ in range(num_blocks)])

        self.out_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, prev_chs in zip(reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1:]))):
                self.out_blocks.append(SparseResBlock3d(prev_chs * 2 if self.use_skip_connection else prev_chs,
                        model_channels, out_channels=chs, upsample=True))
                self.out_blocks.extend([SparseResBlock3d(chs * 2 if self.use_skip_connection else chs,
                        model_channels, out_channels=chs) for _ in range(num_io_res_blocks-1)])
            
        self.out_layer = sp.SparseLinear(model_channels if io_block_channels is None else io_block_channels[0], out_channels)
        self.initialize_weights()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.input_layer(x)

        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h = block(h)
            skips.append(h.feats)
        
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(h.dtype)
        for block in self.blocks:
            h = block(h)

        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)))
            else:
                h = block(h)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))
        return h
