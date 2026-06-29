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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = encoder(batch_ss, sample_posterior=False)
            pred_vector, pred_logits = decoder(z) #[bs, 3, 64, 64, 64], [bs, 1, 64, 64, 64]
            ss_pr = (pred_logits > 0).long()
            pred_grid = ss_pr.squeeze(1) #[bs, 64, 64, 64]

            B, H, W, D = pred_grid.shape

            pred_vector = pred_vector * ss_pr / 32 # [B, 3, 64, 64, 64]
            vector_norm = torch.norm(pred_vector, p=2, dim=1)  # [B, 64, 64, 64]
            non_zero_mask = vector_norm > 0.1  ## usually background

            # ratio_validness = (non_zero_mask.reshape(batch_ss.shape[0], -1).sum(-1) / batch_ss.squeeze(1).reshape(batch_ss.shape[0], -1).sum(-1)) > 0.5

            batch_center_num, batch_remask, mask_score = [], [], []
            for i in range(batch_ss.shape[0]):
                if non_zero_mask[i].sum() > mask_min:
                    obj_indices = torch.argwhere(non_zero_mask[i])
                    obj_coords = (obj_indices.float() + 0.5) / torch.tensor([H, W, D], dtype=torch.float32, device=obj_indices.device) - 0.5

                    # obj_indices_comp = torch.argwhere(pred_grid[i])
                    # obj_coords_comp = (obj_indices_comp.float() + 0.5) / torch.tensor([H, W, D], dtype=torch.float32, device=obj_indices_comp.device) - 0.5

                    pre_obj_vectors = pred_vector[i][:, obj_indices[:, 0], obj_indices[:, 1], obj_indices[:, 2]]
                    moved_coords = obj_coords + pre_obj_vectors.T
                    ##
                    k = int(len(moved_coords) * 0.3)
                    if k < 50:
                        k = 50
                    clustering = DBSCAN(eps=0.1, min_samples=k, n_jobs=12).fit(moved_coords.cpu().numpy())
                    pred_labels = clustering.labels_
                    center_num = len(np.unique(pred_labels[pred_labels != -1]))
                    if center_num ==1:
                        pred_inmask_coords = obj_coords[pred_labels == 0]
                        # pred_inmask_coords = obj_coords_comp
                        scene_pc, center, scale, mask = points[i]
                        dist = torch.cdist(scene_pc, pred_inmask_coords*scale+center)
                        pred_mask = dist.min(1).values < 0.1
                        batch_remask.append(pred_mask[mask])
                        mask_score.append((pred_labels==0).sum()/len(pred_labels))
                    else:
                        batch_remask.append(None)
                        mask_score.append(-1)
                else:
                    center_num = 0
                    batch_remask.append(None)
                    mask_score.append(-1)
                batch_center_num.append(center_num)#*ratio_validness[i])
            return batch_center_num, batch_remask, mask_score