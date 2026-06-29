import torch
import numpy as np
import open3d as o3d
import warnings
warnings.filterwarnings("ignore")
import spconv.pytorch as spconv
from torch.cuda.amp import autocast

# def save_voxel_point_cloud(grid: np.ndarray, out_ply: str) -> np.ndarray:
#     H, W, D = grid.shape
#     idx = np.argwhere(grid > 0)
#     pts = (idx.astype(np.float32) + 0.5) / np.array([H, W, D], np.float32)[None] - 0.5
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts)
#     o3d.io.write_point_cloud(out_ply, pcd)
#     print(f"[saved] point cloud → {out_ply}")
#     return pts

def load_and_normalize_point_cloud(pts):
    center = (pts.max(0).values + pts.min(0).values) / 2.0
    scale = (pts.max(0).values - pts.min(0).values).max()
    pts = (pts - center) / scale  ## making range to (-0.5, 0.5)
    return pts

def convert_points_to_in_field(points):
    points = load_and_normalize_point_cloud(points)
    grid = ((points + 0.5) * 64).long()
    grid = torch.clamp(grid, 0, 63)
    unq_grid = torch.unique(grid, dim=0)
    # feats = torch.ones_like(unq_grid)[:, 0][:, None].float()
    return unq_grid#, feats

def compute_sem_iou(batch_size, in_field, cond_encoder, flow_model, decoder, sampler):
    with autocast():
        cond, fullres = cond_encoder(in_field)
        cond_feat = cond.features
    ##
    batch_dense_grid = []
    for b in range(batch_size):  ## here, I'm not sure whether unique is necessary?
        dense_grid = torch.zeros((16, 16, 16, cond_feat.shape[-1]), device=cond_feat.device, dtype=cond_feat.dtype)
        bs_mask = torch.where(cond.indices[:, 0] == b)[0]
        bs_grid, bs_feat = cond.indices[:, 1:][bs_mask], cond.features[bs_mask]
        dense_grid[bs_grid[:, 0], bs_grid[:, 1], bs_grid[:, 2]] = bs_feat
        batch_dense_grid.append(dense_grid[None, ...])
    cond_feat = torch.cat(batch_dense_grid).permute(0, 4, 1, 2, 3)  ## [bs, C, 16, 16, 16]
    ######################################################################################################
    reso = flow_model.resolution
    noise = torch.randn(batch_size, flow_model.in_channels, reso, reso, reso, device="cuda", dtype=torch.float16)
    neg_cond_tensor = torch.zeros_like(cond_feat)
    with autocast():
        pred = sampler.sample(model=flow_model, noise=noise, cond=cond_feat, neg_cond=neg_cond_tensor,
            steps=20, rescale_t=3.0, cfg_strength=5.0, verbose=True)
        logits = decoder(pred)
        ss_pr = (logits > 0).long() #[bs, 1, 64, 64, 64
        pred_grid = ss_pr.squeeze(1)#[bs, 64, 64, 64

    '''second'''
    _, H, W, D = pred_grid.shape
    # batch_grid, batch_feats = [], []
    batch_grid = []
    non_zero_indication = pred_grid.reshape(batch_size, -1).sum(-1)>0
    batch_counter = 0
    for i in range(batch_size):
        if non_zero_indication[i]:
            idx = torch.argwhere(pred_grid[i] > 0)
            pts = (idx.half() + 0.5) / torch.tensor([H, W, D], device=idx.device)[None] - 0.5
            unq_grid = convert_points_to_in_field(pts)
            # print(pts.shape, pts.min(0).values, pts.max(0).values)
            batch_grid.append(torch.cat((torch.full((unq_grid.shape[0], 1), batch_counter).long().cuda(), unq_grid), dim=-1))
            batch_counter+=1
        # batch_feats.append(feats)
    # batch_grid, batch_feats = torch.cat(batch_grid, dim=0), torch.cat(batch_feats, dim=0)
    batch_grid = torch.cat(batch_grid, dim=0)
    batch_feats = torch.ones((batch_grid.shape[0], 1), device=batch_grid.device).half()
    sparse_shape = list(batch_grid.max(0)[0] + 32)[1:]
    in_field = spconv.SparseConvTensor(features=batch_feats, indices=batch_grid.int().contiguous(),
                                       spatial_shape=sparse_shape, batch_size=batch_size)
    ###################################################################################################################
    cond, fullres = cond_encoder(in_field)
    cond_feat = cond.features
    ##
    batch_dense_grid = []
    for b in range(batch_counter):  ## here, I'm not sure whether unique is necessary?
        dense_grid = torch.zeros((16, 16, 16, cond_feat.shape[-1]), device=cond_feat.device, dtype=cond_feat.dtype)
        bs_mask = torch.where(cond.indices[:, 0] == b)[0]
        bs_grid, bs_feat = cond.indices[:, 1:][bs_mask], cond.features[bs_mask]
        dense_grid[bs_grid[:, 0], bs_grid[:, 1], bs_grid[:, 2]] = bs_feat
        batch_dense_grid.append(dense_grid[None, ...])
    cond_feat = torch.cat(batch_dense_grid).permute(0, 4, 1, 2, 3)  ## [bs, C, 16, 16, 16]
    noise = torch.randn(batch_counter, flow_model.in_channels, reso, reso, reso, device="cuda", dtype=torch.float16)
    neg_cond_tensor = torch.zeros_like(cond_feat)
    ######################################################################################################
    with autocast():
        pred2 = sampler.sample(model=flow_model, noise=noise, cond=cond_feat, neg_cond=neg_cond_tensor,
            steps=20, rescale_t=3.0, cfg_strength=5.0, verbose=True)
        logits2 = decoder(pred2)
        ss_pr2 = (logits2 > 0).long() #[bs, 1, 64, 64, 64
        pred_grid2 = ss_pr2.squeeze(1)#[bs, 64, 64, 64

    pred_grid2[:, :, :, :3] = 0
    pred_grid[:, :, :, :3] = 0
    pred_grid, pred_grid2 = pred_grid.reshape(batch_size, -1), pred_grid2.reshape(batch_counter, -1)
    tmp_pred_grid2 = torch.zeros_like(pred_grid)
    tmp_pred_grid2[torch.where(non_zero_indication)[0]] = pred_grid2
    pred_grid2 = tmp_pred_grid2
    iou = (pred_grid2 * pred_grid).sum(-1) / (pred_grid.sum(-1)+pred_grid2.sum(-1) - (pred_grid2 * pred_grid).sum(-1)+1e-8)
    return iou