import torch
import numpy as np
import time
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor
mask_min = 50

_DBSCAN_POOL = None

def _get_dbscan_pool():
    global _DBSCAN_POOL
    if _DBSCAN_POOL is None:
        _DBSCAN_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 8)
    return _DBSCAN_POOL


def _run_dbscan(moved_coords, k):
    clustering = DBSCAN(eps=0.05, min_samples=k, n_jobs=1).fit(moved_coords)
    pred_labels = clustering.labels_
    center_num = len(np.unique(pred_labels[pred_labels != -1]))
    score = (pred_labels == 0).sum() / len(pred_labels) if center_num == 1 else -1
    return center_num, score


def _resolve_dbscan_jobs():
    env_jobs = os.getenv("CF_DBSCAN_JOBS")
    if env_jobs is not None:
        try:
            return max(1, int(env_jobs))
        except ValueError:
            pass
    return 1

def normalize_point_cloud(pts):
    center = (pts.max(0).values + pts.min(0).values) / 2.0
    scale = (pts.max(0).values - pts.min(0).values).max()
    pts = (pts - center) / scale  ## making range to (-0.5, 0.5)
    return pts, center, scale

def convert_point2ss(pts, voxel_size: float = 1.0 / 32, resolution: int = 32):
    pts01 = pts + 0.5
    idx = torch.floor(pts01 / voxel_size).type(torch.long)
    idx = torch.clamp(idx, 0, resolution - 1)
    grid = torch.zeros((resolution, resolution, resolution), dtype=torch.uint8, device=pts.device)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return grid.unsqueeze(0) # [1, R, R, R]

def compute_cf(batch_ss, encoder, decoder, batch_query, batch_q_len, batch_scene, return_dbscan_time=False, parallel_dbscan=True):
    batch_center_num, mask_score = [], []
    batch_remask = []
    dbscan_time_total = 0.0
    batch_size = batch_ss.shape[0]
    acc_num = 0
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = encoder(batch_ss.cuda().bfloat16())
            batch_cf = decoder(torch.cat(batch_query).bfloat16(), z, batch_q_len).float()

            # Prepare DBSCAN inputs
            dbscan_tasks = []  # (index, moved_coords, k)
            valid_flags = []   # True if we should run DBSCAN for this sample
            for b in range(batch_size):
                curr_pred_cf = batch_cf[acc_num: acc_num + batch_q_len[b]]
                acc_num += batch_q_len[b]
                cf_norm = torch.norm(curr_pred_cf, p=2, dim=1)

                if (cf_norm > 0.1).sum() > mask_min:
                    curr_query = batch_query[b].float().cpu().numpy()
                    moved_coords = curr_query - curr_pred_cf.float().cpu().numpy()
                    k = max(int(len(moved_coords) * 0.3), 50)
                    dbscan_tasks.append((b, moved_coords, k))
                    valid_flags.append(True)
                else:
                    valid_flags.append(False)

            # Run DBSCAN
            if dbscan_tasks:
                t0 = time.perf_counter()
                if parallel_dbscan:
                    pool = _get_dbscan_pool()
                    futures = [pool.submit(_run_dbscan, mc, k) for (_, mc, k) in dbscan_tasks]
                    results = [f.result() for f in futures]
                else:
                    results = [_run_dbscan(mc, k) for (_, mc, k) in dbscan_tasks]
                dbscan_time_total = time.perf_counter() - t0
            else:
                results = []

            # Collect results
            task_idx = 0
            for b in range(batch_size):
                if valid_flags[b]:
                    center_num, score = results[task_idx]
                    task_idx += 1
                    batch_center_num.append(center_num)
                    mask_score.append(score)
                else:
                    batch_center_num.append(0)
                    mask_score.append(-1)

            if return_dbscan_time:
                return batch_center_num, batch_remask, mask_score, dbscan_time_total
            return batch_center_num, batch_remask, mask_score