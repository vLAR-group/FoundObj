from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (sp.SparseTensor): [..., N, D] tensor of queries
            k (sp.SparseTensor): [..., N, D] tensor of keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))
        
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
                torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device)
            )], dim=-1)
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed
    

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0 # 1024% 16
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"#'full'
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None, l_pos_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape #4 x4096 x 1024
        if self._type == "self":
            qkv = self.to_qkv(x) # x torch.Size([4, 4096, 1024])  qkv torch.Size([4, 4096, 3072])
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1) #qkv torch.Size([4, 4096, 3, 16, 64])
            if self.use_rope:
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                if self.qk_rms_norm: #true 对 每个头 的 Q 和 K 向量做 RMSNorm， 可以提升注意力收敛性，避免某些 Token 的 Q/K 过大，影响 softmax 分布，提高注意力机制的稳定性和训练效率
                    q, k, v = qkv.unbind(dim=2)
                    q = self.q_rms_norm(q) #torch.Size([4, 4096, 16, 64])
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v) #内置函数， softmat(Q*K^T / sqart(d_k)) * V
                else:
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else: #cross attention
            Lkv = context.shape[1]  # Lkv 1374
            q = self.to_q(x)  # q torch.Size([4, 4096, 1024])
            kv = self.to_kv(context)  # kv torch.Size([4, 1374, 2048])
            k, v = kv.chunk(2, dim=-1)  # Split kv into k and v along the last dimension
            k = k + (l_pos_emb if l_pos_emb is not None else 0)
            q = q.reshape(B, L, self.num_heads, -1)  # torch.Size([4, 4096, 16, 64])
            k = k.reshape(B, Lkv, self.num_heads, -1)  # torch.Size([4, 1374, 16, 64])
            v = v.reshape(B, Lkv, self.num_heads, -1)  # torch.Size([4, 1374, 16, 64])
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            h = scaled_dot_product_attention(q, k, v)
        h = h.reshape(B, L, -1) # h torch.Size([4, 4096, 16, 64]) -> torch.Size([4, 4096, 1024])
        h = self.to_out(h)# 1024->1024
        return h # h torch.Size([4, 4096, 1024])
