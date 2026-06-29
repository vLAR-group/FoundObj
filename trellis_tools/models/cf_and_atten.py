from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from trellis_tools.modules.norm import GroupNorm32, ChannelLayerNorm32
from trellis_tools.modules.utils import zero_module
from trellis_tools.modules.transformer.blocks import TransformerBlock
import xformers.ops as xops
from trellis_tools.modules.attention.modules import MultiHeadRMSNorm


class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs: int = 6, logspace: bool = True, input_dim: int = 3, include_input: bool = True,
                 include_pi: bool = True) -> None:
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1],
                                                                        -1)  # （N, 3） -> (N, 3, 1) -> (N, 3 , num_freqs) -> (N, 3 * num_freqs)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


def sparse_scaled_dot_product_attention(q, k, v, q_seqlen):
    B, L, H, C = k.shape
    kv_seqlen = [L] * B
    k, v = k.reshape(B * L, H, C), v.reshape(B * L, H, C)

    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
    out = xops.memory_efficient_attention(q, k, v, mask)[0]
    return out.reshape(out.shape[0], -1)


def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    """
    Return a normalization layer.
    """
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None,
                 norm_type: Literal["group", "layer"] = "layer"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3,
                                           padding=1))  ##zero_module 是一种初始化技巧，用于让残差分支一开始 不会产生额外影响（输出为0），从而使网络更稳定、更容易收敛
        self.skip_connection = nn.Conv3d(channels, self.out_channels,
                                         1) if channels != self.out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class DownsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: Literal["conv", "avgpool"] = "conv"):
        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)  # kernal size 2
        elif mode == "avgpool":
            assert in_channels == out_channels, "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)


class SparseStructureEncoder(nn.Module):
    """
    Encoder for Sparse Structure (\mathcal{E}_S in the paper Sec. 3.3).

    Args:
        in_channels (int): Channels of the input.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the encoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """

    def __init__(
            self,
            in_channels: int,
            latent_channels: int,
            num_res_blocks: int,
            channels: List[int],
            num_res_blocks_middle: int = 2,
            norm_type: Literal["group", "layer"] = "layer"):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type

        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)  # 32, kernal size 3

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([ResBlock3d(ch, ch) for _ in range(num_res_blocks)])
            if i < len(channels) - 1:
                self.blocks.append(DownsampleBlock3d(ch, channels[i + 1]))
        self.middle_block = nn.Sequential(
            *[ResBlock3d(channels[-1], channels[-1]) for _ in range(num_res_blocks_middle)])

        self.out_layer = nn.Sequential(norm_layer(norm_type, channels[-1]), nn.SiLU(),
                                       nn.Conv3d(channels[-1], latent_channels * 2, 3, padding=1))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)  # h torch.Size([48, 32, 64, 64, 64])
        for block in self.blocks:
            h = block(h)  # torch.Size([48, 32, 64, 64, 64]), torch.Size([48, 128, 32, 32, 32]), torch.Size([48, 512, 16, 16, 16])
        h = self.middle_block(h)
        return h


class CrossAttention(nn.Module):
    def __init__(self, channels, num_heads, qk_norm=True):
        super().__init__()
        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.to_q = nn.Linear(channels, channels)
        self.to_kv = nn.Linear(channels, channels * 2)

        self.q_rms_norm = MultiHeadRMSNorm(channels // num_heads, num_heads)
        self.k_rms_norm = MultiHeadRMSNorm(channels // num_heads, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, q, kv, q_seqlen):  # q: [L_q, C], kv:[B, L, C]
        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=-1)
        if self.qk_norm:
            q = q.reshape(q.shape[0], self.num_heads, -1)  # [B*L, num_heads, C//num_heads]
            k = k.reshape(k.shape[0], k.shape[1], self.num_heads, -1)  # [B, L, num_heads, C//num_heads]
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
        h = sparse_scaled_dot_product_attention(q, k, v, q_seqlen)
        out = self.to_out(h)
        return out  # [B*L, C]


class CenterField(nn.Module):
    def __init__(self, latent_channels, num_atten_blocks=2, num_heads=8, resolution=32 / 4, out_channels=3):
        super().__init__()

        self.input_layer = nn.Linear(latent_channels, latent_channels)

        self.self_atten_blocks = nn.Sequential(*[TransformerBlock(latent_channels, num_heads=num_heads)
                                                 for _ in range(num_atten_blocks)])

        self.cross_atten = CrossAttention(channels=latent_channels, num_heads=num_heads)

        self.out_layer = nn.Sequential(norm_layer("layer", latent_channels), nn.SiLU(),
                                       nn.Linear(latent_channels, out_channels))
        # self.out_layer = nn.Sequential(nn.Linear(latent_channels, latent_channels), nn.SiLU(),
        #     nn.Linear(latent_channels, 3))

        self.pos_embedder = FourierEmbedder()  # AbsolutePositionEmbedder(latent_channels, 3)
        l_coords = torch.meshgrid(*[torch.arange(res) for res in [resolution] * 3], indexing='ij')
        l_coords = torch.stack(l_coords, dim=-1).reshape(-1, 3)
        l_coords = (l_coords.float() + 0.5) / resolution - 0.5
        l_pos_emb = self.pos_embedder(l_coords)
        self.register_buffer("l_pos_emb", l_pos_emb)
        self.l_pos_proj = nn.Linear(self.pos_embedder.out_dim, latent_channels)
        self.q_pos_proj = nn.Linear(self.pos_embedder.out_dim, latent_channels)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, query, latent, q_lenseq):  ## query need to be [-0.5, 0.5]
        l_pos = self.l_pos_proj(self.l_pos_emb.to(latent.dtype))
        query_pos = self.q_pos_proj(self.pos_embedder(query).to(query.dtype))

        # h = patchify(latent, self.patch_size)
        h = latent
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()  # [bs, num_patches, latent_channels]
        h = self.input_layer(h)
        h = h + l_pos[None, ...]
        for block in self.self_atten_blocks:
            h = block(h)

        # h = h + l_pos[None, ...]
        q = self.cross_atten(query_pos, h, q_lenseq)  # [B*L, C]
        return self.out_layer(q)


class Field(nn.Module):
    def __init__(self, latent_channels, num_atten_blocks=2, num_heads=8, resolution=32 / 4):
        super().__init__()
        self.input_layer = nn.Linear(latent_channels, latent_channels)
        self.self_atten_blocks = nn.Sequential(*[TransformerBlock(latent_channels, num_heads=num_heads)
                                                 for _ in range(num_atten_blocks)])

        self.cf_cross_atten = CrossAttention(channels=latent_channels, num_heads=num_heads)
        self.sdf_cross_atten = CrossAttention(channels=latent_channels, num_heads=num_heads)

        self.cf_out_layer = nn.Sequential(norm_layer("layer", latent_channels), nn.SiLU(),
                                          nn.Linear(latent_channels, 3))
        self.sdf_out_layer = nn.Sequential(norm_layer("layer", latent_channels), nn.SiLU(),
                                           nn.Linear(latent_channels, 1))

        self.pos_embedder = FourierEmbedder()
        l_coords = torch.meshgrid(*[torch.arange(res) for res in [resolution] * 3], indexing='ij')
        l_coords = torch.stack(l_coords, dim=-1).reshape(-1, 3)
        l_coords = (l_coords.float() + 0.5) / resolution - 0.5
        l_pos_emb = self.pos_embedder(l_coords)
        self.register_buffer("l_pos_emb", l_pos_emb)
        self.l_pos_proj = nn.Linear(self.pos_embedder.out_dim, latent_channels)
        self.q_pos_proj = nn.Linear(self.pos_embedder.out_dim, latent_channels)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_latent(self, latent):
        l_pos = self.l_pos_proj(self.l_pos_emb.to(latent.dtype))

        h = latent
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()  # [bs, num_patches, latent_channels]
        h = self.input_layer(h)
        h = h + l_pos[None, ...]
        for block in self.self_atten_blocks:
            h = block(h)
        return h

    def get_cf(self, h, cf_query, cf_q_lenseq):
        cf_query_pos = self.q_pos_proj(self.pos_embedder(cf_query).to(cf_query.dtype))
        cf = self.cf_cross_atten(cf_query_pos, h, cf_q_lenseq)  # [B*L, C]
        return self.cf_out_layer(cf)

    def get_sdf(self, h, sdf_query, sdf_q_lenseq):
        sdf_query_pos = self.q_pos_proj(self.pos_embedder(sdf_query).to(sdf_query.dtype))
        sdf = self.sdf_cross_atten(sdf_query_pos, h, sdf_q_lenseq)  # [B*L, C]
        return self.sdf_out_layer(sdf)

    def forward(self, latent, cf_query, cf_q_lenseq, sdf_query, sdf_q_lenseq):
        h = self.get_latent(latent)
        cf = self.get_cf(h, cf_query, cf_q_lenseq)
        sdf = self.get_sdf(h, sdf_query, sdf_q_lenseq)
        return cf, sdf