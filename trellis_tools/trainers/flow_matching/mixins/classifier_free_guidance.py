import torch
import numpy as np
from trellis_tools.utils.general_utils import dict_foreach


class ClassifierFreeGuidanceMixin:
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond

    def get_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance" 

        if self.p_uncond > 0:#0.1
            # randomly drop the class label
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    return cond.shape[0]
                elif isinstance(cond, list):
                    return len(cond)
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
                
            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]#torch.Size([8, 1374, 1024]), 同cond
            B = get_batch_size(ref_cond)#8
            
            def select(cond, neg_cond, mask):
                if isinstance(cond, torch.Tensor):
                    mask = torch.tensor(mask, device=cond.device).reshape(-1, *[1] * (cond.ndim - 1))#torch.Size([8, 1, 1])
                    return torch.where(mask, neg_cond, cond)#是一个条件选择函数：对 mask 为 True 的位置选择 neg_cond，否则选 cond
                elif isinstance(cond, list):
                    return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")
            
            mask = list(np.random.rand(B) < self.p_uncond)#8 , 全flase， np.random.rand(B)生成一个长度为 B 的数组，其中每个元素是从 [0, 1) 均匀分布中随机采样的浮点数
            if not isinstance(cond, dict):
                cond = select(cond, neg_cond, mask)
            else:
                cond = dict_foreach([cond, neg_cond], lambda x: select(x[0], x[1], mask))
    
        return cond

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for inference.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {'cond': cond, 'neg_cond': neg_cond, **kwargs}

