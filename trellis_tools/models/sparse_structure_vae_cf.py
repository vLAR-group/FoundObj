from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from trellis_tools.modules.norm import GroupNorm32, ChannelLayerNorm32
from trellis_tools.modules.spatial import pixel_shuffle_3d
from trellis_tools.modules.utils import zero_module
from trellis_tools.modules.transformer import AbsolutePositionEmbedder
from einops import rearrange


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
    def __init__(self,channels: int, out_channels: Optional[int] = None, norm_type: Literal["group", "layer"] = "layer"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))##zero_module 是一种初始化技巧，用于让残差分支一开始 不会产生额外影响（输出为0），从而使网络更稳定、更容易收敛
        self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()

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
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2) #kernal size 2
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
            self.conv = nn.Conv3d(in_channels, out_channels*8, 3, padding=1)
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

        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)#32, kernal size 3

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([ResBlock3d(ch, ch) for _ in range(num_res_blocks)])
            if i < len(channels) - 1:
                self.blocks.append(DownsampleBlock3d(ch, channels[i+1]))
        
        self.middle_block = nn.Sequential(*[ResBlock3d(channels[-1], channels[-1]) for _ in range(num_res_blocks_middle)])

        # self.out_layer = nn.Sequential(norm_layer(norm_type, channels[-1]), nn.SiLU(),
        #     nn.Conv3d(channels[-1], latent_channels*2, 3, padding=1))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False) -> torch.Tensor:
        h = self.input_layer(x) # h torch.Size([48, 32, 64, 64, 64])

        for block in self.blocks:
            h = block(h)#torch.Size([48, 32, 64, 64, 64]), torch.Size([48, 128, 32, 32, 32]), torch.Size([48, 512, 16, 16, 16])
        h = self.middle_block(h)
        return h
        # h = self.out_layer(h)#torch.Size([48, 16, 16, 16, 16])
        #
        # mean, logvar = h.chunk(2, dim=1)#torch.Size([48, 8, 16, 16, 16])
        #
        # if sample_posterior:
        #     std = torch.exp(0.5 * logvar)
        #     z = mean + std * torch.randn_like(std)
        # else:
        #     z = mean
        #
        # if return_raw:
        #     return z, mean, logvar
        # return z


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class MLP(nn.Module):
    def __init__(
            self, *,
            width: int,
            expand_ratio: int = 4,
            output_width: int = None,
            drop_path_rate: float = 0.0):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width if output_width is not None else width)
        self.gelu = nn.GELU(approximate='tanh')
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self,
            *,
            heads: int,
            n_data: Optional[int] = None,
            width=None,
            qk_norm=False,
            norm_layer=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, q, k, v):
        _, n_ctx, _ = q.shape
        bs, n_data, width = k.shape
        q = q.view(bs, n_ctx, self.heads, -1)
        k = k.view(bs, n_data, self.heads, -1)
        v = v.view(bs, n_data, self.heads, -1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(self,
            *,
            width: int,
            heads: int,
            qkv_bias: bool = True,
            n_data: Optional[int] = None,
            data_width: Optional[int] = None,
            norm_layer=nn.LayerNorm,
            qk_norm: bool = False):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_k = nn.Linear(self.data_width, width, bias=qkv_bias)
        self.c_v = nn.Linear(self.data_width, width, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads,
            n_data=n_data,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm)

    def forward(self, x, k, v):
        x = self.c_q(x)
        k = self.c_k(k)
        v = self.c_v(v)
        x = self.attention(x, k, v)
        x = self.c_proj(x)
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self,
            *,
            n_data: Optional[int] = None,
            width: int,
            heads: int,
            mlp_expand_ratio: int = 4,
            data_width: Optional[int] = None,
            qkv_bias: bool = True,  # false
            norm_layer=nn.LayerNorm,
            qk_norm: bool = False):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = norm_layer(data_width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        x = x + self.attn(self.ln_1(x.float()).to(dtype=torch.float16), self.ln_2(k.float()).to(dtype=torch.float16),
                          self.ln_2(v.float()).to(dtype=torch.float16))
        x = x + self.mlp(self.ln_3(x.float()).to(dtype=torch.float16))
        return x


class CrossAttentionDecoder(nn.Module):
    def __init__(self,
            *,
            num_latents: int = 4096,
            out_channels: int = 1,
            l_resolution: int = 64,
            width: int = 128,
            heads: int = 16,
            mlp_expand_ratio: int = 4,
            latent_dim: int = 32,
            enable_ln_post: bool = True,
            qkv_bias: bool = True,
            qk_norm: bool = False):
        super().__init__()

        self.l_resolution = l_resolution
        self.enable_ln_post = enable_ln_post
        self.latent_dim = latent_dim
        self.query_pos_proj = nn.Linear(width, width)
        if self.latent_dim != 1:
            self.latents_proj = nn.Linear(self.latent_dim, width)  # 32 ->128
        if self.enable_ln_post == False:
            qk_norm = False
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            mlp_expand_ratio=mlp_expand_ratio,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm)

        if self.enable_ln_post:
            self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)

        self.pos_embedder = AbsolutePositionEmbedder(width, 3)

        ##这是体素坐标，跟query不匹配， 转换为点云坐标（小数点）
        l_coords = torch.meshgrid(*[torch.arange(res) for res in [l_resolution] * 3], indexing='ij')# resolution = 128。####meshgrid查一下indexing='ij'， 格子点顺序是怎样的
        l_coords = torch.stack(l_coords, dim=-1).reshape(-1, 3) #3x64x64x64 （16x16x16对应格子的X,Y,Z坐标） -> 4096x3 （16x16x16=4096, 4096个格子对应的坐标）
        l_coords = (l_coords.float() + 0.5) / l_resolution - 0.5
        l_pos_emb = self.pos_embedder(l_coords) #64**3 x3-> 64**3 x512
        self.register_buffer("l_pos_emb", l_pos_emb)
        self.key_pos_proj = nn.Linear(width, width)

    def forward(self, queries=None, latents=None):
        target_dtype = queries.dtype
        bs, num_points = queries.shape[0], queries.shape[1]

        query_pos_embeddings = self.pos_embedder(queries.reshape(bs * num_points, 3)).to(dtype=target_dtype)  # num_points x 3 -> 128**3 x 512
        query_pos_embeddings = query_pos_embeddings.reshape(bs, num_points, -1)  # bs x num_points x dim
        query_embeddings = self.query_pos_proj(query_pos_embeddings)#(bs, num_points, -1)

        latents = self.latents_proj(latents)  # 4096x32 -> 4096x128
        k = latents + self.key_pos_proj(self.l_pos_emb.to(dtype = target_dtype)[None, ...])
        v = latents
        x = self.cross_attn_decoder(query_embeddings, k, v).to(dtype=target_dtype)
        if self.enable_ln_post:
            x = self.ln_post(x.float()).to(dtype=target_dtype)
        occ = self.output_proj(x)
        return occ


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
        out_channels_cf: int,
        out_channels_ss: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer"):
        super().__init__()
        self.out_channels_cf = out_channels_cf
        self.out_channels_ss = out_channels_ss
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(*[ResBlock3d(channels[0], channels[0])
            for _ in range(num_res_blocks_middle)])

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([ResBlock3d(ch, ch) for _ in range(num_res_blocks)])
            if i < len(channels) - 1:
                self.blocks.append(UpsampleBlock3d(ch, channels[i+1]))

        self.out_layer_cf = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels_cf, 3, padding=1))

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels_ss, 3, padding=1))


    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)  # h torch.Size([48, 512, 16, 16, 16])

        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)  # torch.Size([48, 128, 32, 32, 32])， torch.Size([4, 32, 64, 64, 64])

        ss = self.out_layer(h)
        cf = self.out_layer_cf(h)
        return cf, ss
