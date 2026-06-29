from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from trellis_tools.modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from trellis_tools.modules.spatial import patchify, unpatchify
from trellis_tools.trainers.flow_matching.mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from safetensors.torch import load_file

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True), #hidden_size = 1024
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
        # wi = 1/ (10000 ** (i / half))  # i = 0, 1, ..., half - 1  高频到低频
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)#128
        args = t[:, None].float() * freqs[None] #4x128 #t 是 (N,)，扩展为 (N, 1)，与 freqs 做广播乘法
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)#4x256
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)#4x256 - > 4x1024
        return t_emb


##############################
class CondDecoder(nn.Module, ClassifierFreeGuidanceMixin):
    def __init__(self, in_channels, out_channels, p_uncond=0.1, pos_emb=None):
        super().__init__()
        self.cond_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.p_uncond = p_uncond
        #self.pos_emb = pos_emb  # [1, 4096, C] 或 None
        self.register_buffer("pos_emb", pos_emb)
            
    def _load_pretrained(self, path: str, keys: List[str]):
        if path.endswith(".safetensors"):
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location='cpu')
            if "model" in state_dict:
                state_dict = state_dict["model"]

        try:
            with torch.no_grad():
                self.cond_conv.weight.data.copy_(state_dict[keys[0]].view_as(self.cond_conv.weight))
                self.cond_conv.bias.data.copy_(state_dict[keys[1]])
                print(f"[DEBUG] CondDecoder loaded from {path}: {keys}")
        except Exception as e:
            print(f"[DEBUG] Failed loading pretrained decoder keys: {e}")
    
    def forward(self, cond_voxel: torch.Tensor, neg_cond: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            cond_voxel: shape (B, C, 16, 16, 16)
            neg_cond: same shape for classifier-free guidance
        Returns:
            cond: shape (B, 4096, C)
        """
        cond = self.cond_conv(cond_voxel)  # (B, C, 16, 16, 16)
        B, C, X, Y, Z = cond.shape
        cond = cond.permute(0, 2, 3, 4, 1).reshape(B, X*Y*Z, C)  # (B, 4096, C)
        # 加位置编码#################
        cond = cond + self.pos_emb[None]

        # Apply classifier-free guidance
        
        if neg_cond is None:
            neg_cond = torch.zeros_like(cond)
                
        cond = self.get_cond(cond, neg_cond=neg_cond)
        
        ########采样时候，零条件可返回全0
        if torch.allclose(cond_voxel, torch.zeros_like(cond_voxel), atol=1e-8):
            cond = torch.zeros_like(cond)

        return cond  


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
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')# resolution = 16, patchsize = 1
            coords = torch.stack(coords, dim=-1).reshape(-1, 3) #3x16x16x16 （16x16x16对应格子的X,Y,Z坐标） -> 4096x3 （16x16x16=4096, 4096个格子对应的坐标）
            pos_emb = pos_embedder(coords)#4096x3->4096x1024 
            #这里的 pos_emb 是位置编码，是固定值（或者提前计算好），将 pos_emb 注册为当前 nn.Module 的一个 buffer，而不是一个可训练的参数（它 不应该参与反向传播和梯度更新（不像参数））
            #它需要保存在模型的 .pt 文件中（state_dict() 会保存 buffer），它应该能自动随着模型 .cuda()、.to()、.eval() 等操作而移动/切换
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels) #in_channels = 8, patch_size = 1, modelchannel=1024 
        
        #####################################

        self.cond_decoder = CondDecoder(
            in_channels=128,#in_channels,
            out_channels=model_channels,
            p_uncond=0.1,
            pos_emb=self.pos_emb)
        
        self.blocks = nn.ModuleList([ModulatedTransformerCrossBlock(
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
        """
        Return the device of the model.
        """
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

    def forward(self, x, t, cond):
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
            h = block(h, t_emb, cond)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)#torch.Size([4, 4096, 8])-> torch.Size([4, 8, 16, 16, 16])
        h = unpatchify(h, self.patch_size).contiguous()

        return h
