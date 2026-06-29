from typing import *
import torch
import numpy as np
from tqdm import tqdm


class FlowEulerSampler:
    """
        sigma_min: The minimum scale of noise in flow.
    """
    def inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float16)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond)


    def get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self.inference_model(model, x_t, t, cond, **kwargs)  # pred_v will be x_0 (noise) - x_1
        return pred_v


    @torch.no_grad()
    def sample_once(self, model, x_t, t: float, t_next: float, cond: Optional[Any] = None, **kwargs):
        pred_v = self.get_model_prediction(model, x_t, t, cond, **kwargs)  # t:np.float64(1.0) t_prev:np.float64(0.98)
        pred_x_next = x_t + (t_next - t) * pred_v
        return pred_x_next


    @torch.no_grad()
    def sample(self, model, noise, cond: Optional[Any] = None, steps: int = 50, rescale_t: float = 1.0, verbose: bool = True, **kwargs):
        # pred_list = []
        pred_x_next = noise
        t_seq = np.linspace(0, 1, steps + 1)  # (26,)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)  # rescaled t 3.0
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))  # 25
        for t, t_next in tqdm(t_pairs, desc="Sampling", disable=not verbose):
        # for t, t_next in t_pairs:
            pred_x_next = self.sample_once(model, pred_x_next, t, t_next, cond, **kwargs)
            # pred_list.append(pred_x_next)
        return pred_x_next#, pred_list



class FlowEulerCfgSampler(FlowEulerSampler):

    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps: int = 50, rescale_t: float = 1.0,
            cfg_strength: float = 3.0, cfg_interval: Tuple[float, float] = (0.0, 1.0), verbose: bool = True):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond,
                              cfg_strength=cfg_strength, cfg_interval=cfg_interval)


    def inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            cond_pred = super().inference_model(model, x_t, t, cond)
            uncond_pred = super().inference_model(model, x_t, t, neg_cond)
            return (1 + cfg_strength) * cond_pred - cfg_strength * uncond_pred
            # return cfg_strength * (cond_pred - uncond_pred) + uncond_pred
        else:
            return super().inference_model(model, x_t, t, cond)
