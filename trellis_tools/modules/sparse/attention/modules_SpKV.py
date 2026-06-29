from typing import *
import torch
from  spconv.pytorch import SparseConvTensor
import torch.nn as nn
import torch.nn.functional as F
from .serialized_attn import SerializeMode, sparse_serialized_scaled_dot_product_self_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from ...attention import RotaryPositionEmbedder
from ..basic import sparse_unbind
import xformers.ops as xops

def sparse_scaled_dot_product_attention(q, k, v, kv_seqlen):
    N, L, H, C = q.shape
    q_seqlen = [L] * N
    q = q.reshape(N * L, H, C)  # [T_Q, H, C]

    # kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]

    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
    out = xops.memory_efficient_attention(q, k, v, mask)[0]
    return out.reshape(N, L, H, -1)

class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[SparseConvTensor, torch.Tensor]) -> Union[SparseConvTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, SparseConvTensor):
            x = x.replace_feature(F.normalize(x.features, dim=-1))
        else:
            x = F.normalize(x, dim=-1)            
        return (x * self.gamma * self.scale).to(x_type)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)


class SparseMultiHeadAttention_SpKV(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "serialized", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            # self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            # self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    @staticmethod
    def _linear(module: nn.Linear, x: Union[SparseConvTensor, torch.Tensor]) -> Union[SparseConvTensor, torch.Tensor]:
        if isinstance(x, SparseConvTensor):
            return x.replace_feature(module(x.features))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x: Union[SparseConvTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[SparseConvTensor, torch.Tensor]:
        if isinstance(x, SparseConvTensor):
            return x.replace_feature(x.features.reshape(*shape))
        else:
            return x.reshape(*x.shape[:2], *shape)

    # def _fused_pre(self, x: Union[SparseConvTensor, torch.Tensor], num_fused: int) -> Union[SparseConvTensor, torch.Tensor]:
    #     if isinstance(x, SparseConvTensor):
    #         x_feats = x.features.unsqueeze(0)
    #     else:
    #         x_feats = x
    #     x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
    #     return x.replace_feature(x_feats.squeeze(0)) if isinstance(x, SparseConvTensor) else x_feats
    def _fused_pre(self, x, num_fused: int):
        return x.reshape(x.shape[0], num_fused, self.num_heads, -1)

    def _rope(self, qkv: SparseConvTensor):
        q, k, v = qkv.features.unbind(dim=1)   # [T, H, C]
        q, k = self.rope(q, k, qkv.indices[:, 1:])
        qkv = qkv.replace_feature(torch.stack([q, k, v], dim=1))
        return qkv
    
    def forward(self, x: torch.Tensor, context, conetex_len):
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.use_rope:
                qkv = self._rope(qkv)
            if self.qk_rms_norm:
                if isinstance(qkv, SparseConvTensor):
                    q, k, v = sparse_unbind(qkv, dim=1)
                else:
                    q, k, v = qkv.unbind(dim=1)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = qkv.replace_feature(torch.stack([q.features, k.features, v.features], dim=1))
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "serialized":
                h = sparse_serialized_scaled_dot_product_self_attention(
                    qkv, self.window_size, serialize_mode=self.serialize_mode, shift_sequence=self.shift_sequence, shift_window=self.shift_window)
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window)
        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            k, v = kv.unbind(dim=1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                # kv = kv.replace_feature(torch.stack([k.features, v.features], dim=1))
            h = sparse_scaled_dot_product_attention(q, k, v, conetex_len)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h
