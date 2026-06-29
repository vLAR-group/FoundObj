import torch


def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    3D pixel shuffle.
    """
    B, C, H, W, D = x.shape # 48, 1024, 16, 16, 16
    C_ = C // scale_factor**3 # 1024 // 2**3 = 128
    x = x.reshape(B, C_, scale_factor, scale_factor, scale_factor, H, W, D)#torch.Size([48, 128, 2, 2, 2, 16, 16, 16])
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)#torch.Size([48, 128, 16, 2, 16, 2, 16, 2])
    x = x.reshape(B, C_, H*scale_factor, W*scale_factor, D*scale_factor)#torch.Size([48, 128, 32, 32, 32])
    return x

#将一个 3D张量（如3D图像或体素网格）按照 patch_size 分块，使每个 patch 成为一个“token”，并重新排列数据，以便后续送入 Transformer 等模型中
def patchify(x: torch.Tensor, patch_size: int):
    """
    Patchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    DIM = x.dim() - 2 # 3
    for d in range(2, DIM + 2):
        assert x.shape[d] % patch_size == 0, f"Dimension {d} of input tensor must be divisible by patch size, got {x.shape[d]} and {patch_size}"
#把每个维度拆成 (num_patches, patch_size)
    x = x.reshape(*x.shape[:2], *sum([[x.shape[d] // patch_size, patch_size] for d in range(2, DIM + 2)], [])) # 4x8x16x1x16x1x16x1
    #2*i+2 是 num_patches 部分, 2*i+3 是 patch_size 部分
    x = x.permute(0, 1, *([2 * i + 3 for i in range(DIM)] + [2 * i + 2 for i in range(DIM)]))#torch.Size([4, 8, 1, 1, 1, 16, 16, 16])
    x = x.reshape(x.shape[0], x.shape[1] * (patch_size ** DIM), *(x.shape[-DIM:]))#torch.Size([4, 8, 16, 16, 16])
    return x


def unpatchify(x: torch.Tensor, patch_size: int):
    """
    Unpatchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    DIM = x.dim() - 2 #DIM 3
    assert x.shape[1] % (patch_size ** DIM) == 0, f"Second dimension of input tensor must be divisible by patch size to unpatchify, got {x.shape[1]} and {patch_size ** DIM}"

    x = x.reshape(x.shape[0], x.shape[1] // (patch_size ** DIM), *([patch_size] * DIM), *(x.shape[-DIM:])) #torch.Size([4, 8, 1, 1, 1, 16, 16, 16])
    x = x.permute(0, 1, *(sum([[2 + DIM + i, 2 + i] for i in range(DIM)], []))) #torch.Size([4, 8, 1, 1, 1, 16, 16, 16])
    x = x.reshape(x.shape[0], x.shape[1], *[x.shape[2 + 2 * i] * patch_size for i in range(DIM)])#torch.Size([4, 8, 16, 1, 16, 1, 16, 1])
    return x#torch.Size([4, 8, 16, 16, 16])
