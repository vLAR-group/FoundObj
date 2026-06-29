from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from modules.sparse.transformer import ModulatedSparseTransformerCrossBlock_SpKV
from modules.spatial import patchify, unpatchify

####################
from trainers.flow_matching.mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from safetensors.torch import load_file

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        #每个时间步 t 被嵌入成一个高维向量，供模型使用，使得模型可以“感知当前在第几步”
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2 #dim 256, half 128
        freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



class CondDecoder(nn.Module, ClassifierFreeGuidanceMixin):
    def __init__(self, in_channels, out_channels, p_uncond=0.1, pos_emb=None):
        super().__init__()
        self.cond_feats_fc = nn.Linear(in_channels, out_channels)
        self.cond_pos_fc = nn.Linear(out_channels, out_channels)
        self.p_uncond = p_uncond
        self.pos_emb = pos_emb

    def forward(self, cond: List, neg_cond: torch.Tensor = None) -> torch.Tensor:
        cond_feats = self.cond_feats_fc(cond[0]) # (B, N, C)
        # pos_emb = self.cond_pos_fc(self.pos_embedder(cond[1]))
        # pos_emb = self.pos_embedder(cond[1])
        i, j, k = cond[1].long().unbind(-1)
        cond_feats = cond_feats + self.pos_emb.to(cond_feats.device)[i, j, k]

        # Apply classifier-free guidance
        
        if neg_cond is None:
            neg_cond = torch.zeros_like(cond_feats)
                
        cond_feats = self.get_cond(cond_feats, neg_cond=neg_cond)
        
        ########采样时候，零条件可返回全0
        if torch.allclose(cond[0], torch.zeros_like(cond[0]), atol=1e-8):
            cond_feats = torch.zeros_like(cond_feats)
        return cond_feats


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(),nn.Linear(model_channels, 6 * model_channels, bias=True))

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)
        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
        #####################################
        self.cond_decoder = CondDecoder(
            in_channels=128,#in_channels,
            out_channels=model_channels,
            p_uncond=0.1, pos_emb=pos_emb.reshape(resolution // patch_size, resolution // patch_size, resolution // patch_size, -1))#pos_embedder=pos_embedder)
            # pos_emb=self.pos_emb)


        self.blocks = nn.ModuleList([ModulatedSparseTransformerCrossBlock_SpKV(
                model_channels,#1024
                cond_channels,#1024
                num_heads=self.num_heads,#16
                mlp_ratio=self.mlp_ratio,#4
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross)
            for _ in range(num_blocks)])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

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

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x, t, cond, kv_seq=None):
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        
        ############################
        cond = self.cond_decoder(cond)
        
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond, kv_seq)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)#torch.Size([4, 4096, 8])-> torch.Size([4, 8, 16, 16, 16])
        h = unpatchify(h, self.patch_size).contiguous()

        return h
