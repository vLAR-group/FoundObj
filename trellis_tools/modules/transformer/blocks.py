from typing import *
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """
    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels #1024
        self.in_channels = in_channels #3
        self.freq_dim = channels // in_channels // 2 # 1024 // 3 // 2 = 170 取整除
        #位置编码中的频率因子， wi = 1 / (10000 ** (i/d))，其中 i 是频率维度的索引，d 是频率维度的大小
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim # i/d 将 0 到 freq_dim - 1 均匀映射到区间 [0, 1)
        self.freqs = 1.0 / (10000 ** self.freqs) # 构造的是一组不同频率的倒数（decaying frequencies）

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)#x:12288  # 外积矩阵 out[i, j] = x[i] * self.freqs[j]
        #“位置信息”x，用不同频率的 sin/cos 组合投影成一个高维向量（340维），用于输入 transformer 等模型，携带结构/位置信息
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1) #out: (12288, 170) #对外积结果进行正弦余弦变换， 每一行从 170 维变成 340 维
        return out #out.shape == (12288, 340)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape #4096x3
        assert D == self.in_channels, "Input dimension must match number of input channels"
        #把“坐标”映射为可被神经网络理解的连续向量编码
        embed = self._sin_cos_embedding(x.reshape(-1))#embed  torch.Size([12288, 340])
        embed = embed.reshape(N, -1) #embed : 4096x1020
        #每一个 3D 空间位置被编码成一个 1020 维向量， 再补齐到 1024 维
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed # embed.shape == (4096, 1024)


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),#中间隐藏层扩大倍数
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[int] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class TransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN).
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )

    def _forward(self, x: torch.Tensor, context: torch.Tensor):
        h = self.norm1(x)
        h = self.self_attn(h)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context, use_reentrant=False)
        else:
            return self._forward(x, context)
        