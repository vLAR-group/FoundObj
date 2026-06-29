from typing import *
import torch
import torch.nn as nn
from modules.norm import GroupNorm32, ChannelLayerNorm32
from modules.spatial import pixel_shuffle_3d
from modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
import torch.nn.functional as F
from modules.transformer import AbsolutePositionEmbedder
from modules.attention import MultiHeadAttention


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
    def __init__(
            self,
            channels: int,
            out_channels: Optional[int] = None,
            norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3,
                                           padding=1))  ##zero_module 是一种初始化技巧，用于让残差分支一开始 不会产生额外影响（输出为0），从而使网络更稳定、更容易收敛
        self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()

    # #输入的通道数 channels 与输出的 self.out_channels 相等，就直接跳过（Identity()），如果不等，就用 Conv3d(1x1x1) 卷积层来调整输入的通道数匹配输出
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


class UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: Literal["conv", "nearest"] = "conv"):
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)
        elif mode == "nearest":
            assert in_channels == out_channels, "Nearest mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


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

    def __init__(self,
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
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)])

            if i < len(channels) - 1:
                self.blocks.append(
                    DownsampleBlock3d(ch, channels[i + 1]))

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1])
            for _ in range(num_res_blocks_middle)])

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels * 2, 3, padding=1))

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False) -> torch.Tensor:
        h = self.input_layer(x)  # h torch.Size([48, 32, 64, 64, 64])

        for block in self.blocks:
            h = block(h)  # torch.Size([48, 32, 64, 64, 64]), torch.Size([48, 128, 32, 32, 32]), torch.Size([48, 512, 16, 16, 16])
        h = self.middle_block(h)

        h = self.out_layer(h)  # torch.Size([48, 16, 16, 16, 16])

        mean, logvar = h.chunk(2, dim=1)  # torch.Size([48, 8, 16, 16, 16])

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean

        if return_raw:
            return z, mean, logvar
        return z


class CrossAttention_trellis(nn.Module):
    def __init__(
            self,
            *,
            l_resolution: int = 16,
            pos_channels: int = 128,
            channels: int = 128,
            ctx_channels: int = 32,
            num_heads: int = 16,
            qkv_bias: bool = True,
            qk_rms_norm_cross: bool = False
    ):
        super().__init__()
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross)
        self.l_resolution = l_resolution
        self.pos_embedder = AbsolutePositionEmbedder(pos_channels, 3)
        self.query_pos_proj = nn.Sequential(nn.Linear(pos_channels, pos_channels), nn.LeakyReLU(), nn.Linear(pos_channels, pos_channels))
        self.key_pos_proj = nn.Sequential(nn.Linear(pos_channels, pos_channels), nn.LeakyReLU(), nn.Linear(pos_channels, pos_channels))

        l_coords = torch.meshgrid(*[torch.arange(res) for res in [l_resolution] * 3], indexing='ij')  # resolution = 128。####meshgrid查一下indexing='ij'， 格子点顺序是怎样的
        l_coords = torch.stack(l_coords, dim=-1).reshape(-1, 3)  # 3x64x64x64 （16x16x16对应格子的X,Y,Z坐标） -> 4096x3 （16x16x16=4096, 4096个格子对应的坐标）
        l_coords = l_coords.float() - l_resolution/2 + 0.5
        l_pos_emb = self.pos_embedder(l_coords)  # 64**3 x3-> 64**3 x512
        self.register_buffer("l_pos_emb", l_pos_emb)

    def forward(self, queries=None, latents=None):
        original_shape = queries.shape
        bs, num_points = original_shape[0], original_shape[1]

        queries = queries*self.l_resolution
        queries = queries.view(bs * num_points, 3)  # bs x num_points x 3 -> (bs * num_points) x 3
        queries = self.pos_embedder(queries)#.to(dtype=queries.dtype)  # num_points x 3 -> 128**3 x 512
        queries = queries.view(bs, num_points, -1)  # bs x num_points x dim
        queries = self.query_pos_proj(queries)#(bs, num_points, -1)

        x = self.cross_attn(queries, latents, l_pos_emb=self.key_pos_proj(self.l_pos_emb.to(dtype=queries.dtype)))  # h 为 q
        # x = self.cross_attn(queries, latents, l_pos_emb=None )
        return x


class SparseStructureDecoder(nn.Module):
    """
    Decoder for Sparse Structure (\mathcal{D}_S in the paper Sec. 3.3).

    Args:
        out_channels (int): Channels of the output.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the decoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """

    def __init__(
            self,
            out_channels: int,
            latent_channels: int,
            num_res_blocks: int,
            channels: List[int],
            num_res_blocks_middle: int = 2,
            norm_type: Literal["group", "layer"] = "layer"):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type

        self.input_layer = nn.Conv3d(latent_channels, channels[2], 3, padding=1)
        self.output_proj = nn.Linear(channels[1], out_channels)
        self.sdf_decoder = CrossAttention_trellis(qk_rms_norm_cross=True)

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)  # h torch.Size([48, 512, 16, 16, 16])

        h = self.sdf_decoder(q, h.flatten(2).permute(0, 2, 1))  # torch.Size([4, 262144, 1])
        h = self.output_proj(h)
        return h
