import torch
import numpy as np
import open3d as o3d
import warnings
warnings.filterwarnings("ignore")
# import spconv.pytorch as spconv
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from torch.cuda.amp import autocast
from lib.helper_ply import write_ply

mask_min = 50

def normalize_point_cloud(pts):
    center = (pts.max(0).values + pts.min(0).values) / 2.0
    scale = (pts.max(0).values - pts.min(0).values).max()
    pts = (pts - center) / scale  ## making range to (-0.5, 0.5)
    # return pts#* 0.75 + 0.25 * torch.rand((1, 3), device=pts.device) - 0.125
    return pts, center, scale

def convert_point2ss(pts, voxel_size: float = 1.0 / 32, resolution: int = 32):
    pts01 = pts + 0.5
    idx = torch.floor(pts01 / voxel_size).type(torch.long)
    idx = torch.clamp(idx, 0, resolution - 1)
    grid = torch.zeros((resolution, resolution, resolution), dtype=torch.uint8, device=pts.device)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return grid.unsqueeze(0) # [1, R, R, R]

def compute_cf(batch_ss, encoder, decoder, points): # points is normalized grids
    with torch.no_grad():
        with autocast():
            z = encoder(batch_ss, sample_posterior=False)
            pred_logits = decoder(z) #[bs, 3, 64, 64, 64], [bs, 1, 64, 64, 64]
            ss_pr = (pred_logits > 0).long()
            pred_grid = ss_pr.squeeze(1) #[bs, 64, 64, 64]

            B, H, W, D = pred_grid.shape

            batch_cd, batch_remask, mask_score = [], [], []
            for i in range(batch_ss.shape[0]):
                if pred_grid[i].sum() > mask_min:
                    obj_indices = torch.argwhere(pred_grid[i])
                    obj_coords = (obj_indices.float() + 0.5) / torch.tensor([H, W, D], dtype=torch.float32, device=obj_indices.device) - 0.5
                    ##
                    scene_pc, center, scale, mask = points[i]
                    dist = torch.cdist(scene_pc, obj_coords * scale + center)

                    pred_mask = dist.min(1).values < 0.1

                    # cd = dist[pred_mask].min(0).values.mean()/2+dist[pred_mask].min(1).values.mean()/2
                    batch_cd.append(0)
                    ###

                    if pred_mask.sum() >mask_min:
                        batch_remask.append(pred_mask[mask])
                        mask_score.append((pred_mask.sum()/len(pred_mask)).cpu().numpy())
                    else:
                        batch_remask.append(None)
                        mask_score.append(-1)
                else:
                    batch_remask.append(None)
                    mask_score.append(-1)
                    batch_cd.append(1000)
            return batch_cd, batch_remask, mask_score