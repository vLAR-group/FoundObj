"""Generate pseudo instance masks via iterative Normalized Cut on superpoint features.

Ported from MyTrellis/playground/ncut.py, using FoundObj superpoints and SP features.

Usage:
    python preprocessing/generate_ncut_masks.py
    python preprocessing/generate_ncut_masks.py --max-scenes 1 --visualize
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch_scatter import scatter_mean, scatter_sum

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from preprocessing.get_neighbors import load_dist_adjacency

voxel_size = 0.02


def load_superpoint_neighbors(load_path: str):
    """Load superpoint neighbor dict from binary file."""
    neighbor_dict = {}
    with open(load_path, "rb") as f:
        num_sp = int.from_bytes(f.read(4), byteorder="little")
        for _ in range(num_sp):
            sp_id = int.from_bytes(f.read(4), byteorder="little")
            num_neighbors = int.from_bytes(f.read(4), byteorder="little")
            neighbor_dict[sp_id] = [int.from_bytes(f.read(4), byteorder="little")
                for _ in range(num_neighbors)]
    return neighbor_dict


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def list_train_scenes(data_root):
    """List scene IDs from processed_aligns/train/."""
    train_dir = os.path.join(data_root, "train")
    scenes = sorted(os.path.splitext(f)[0] for f in os.listdir(train_dir) if f.endswith(".npy"))
    return scenes


def voxelize(coords):
    scale = 1 / voxel_size
    coords = coords - coords.min(0)
    coords = np.floor(coords * scale)
    coords, unique_map, inverse_map = np.unique(coords, return_index=True, return_inverse=True, axis=0)
    return coords, unique_map, inverse_map


# ──────────────────────────────────────────────────────────────────────────────
# NCut algorithm (from MyTrellis/playground/unscene3d.py)
# ──────────────────────────────────────────────────────────────────────────────

def cosine_sim(feats_k, feats_q):
    eps = 1e-10
    key_feats = feats_k / (feats_k.norm(dim=1, keepdim=True) + eps)
    queries = feats_q / (feats_q.norm(dim=1, keepdim=True) + eps)
    attn = queries @ key_feats.T
    attn -= attn.min(-1, keepdim=True)[0]
    attn /= attn.max(-1, keepdim=True)[0] + eps
    return attn


def normalize_mat(A, eps=1e-5):
    A -= np.min(A[np.nonzero(A)]) if np.any(A > 0) else 0
    A[A < 0] = 0.0
    A /= A.max() + eps
    return A


def get_affinity_matrix(feats, tau=0.15, eps=1e-5, connec_mask=None):
    feats_a = F.normalize(feats, p=2, dim=-1)
    A = cosine_sim(feats_a, feats_a)
    A = A.cpu().numpy()
    A = normalize_mat(A)

    if connec_mask is not None:
        A = A * connec_mask

    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=0)
    D = np.diag(d_i)
    return A, D


def get_masked_affinity_matrix(painting, feats, mask):
    num_segment, dim = feats.shape
    painting = painting.view(num_segment, 1) + mask.view(num_segment, 1)
    painting[painting > 0] = 1
    painting[painting <= 0] = 0
    feats = (1 - painting) * feats.clone()
    return feats, painting.squeeze()


def second_smallest_eigenvector(A, D):
    A = torch.from_numpy(A).cuda().double()
    D = torch.sum(A, dim=0)
    D_diag = torch.diag_embed(D)
    D_over_sqrt = torch.diag_embed(torch.sqrt(1.0 / D))
    L = torch.matmul(D_over_sqrt, torch.matmul(D_diag - A, D_over_sqrt))
    eigenvalues, eigenvectors = torch.linalg.eigh(L, UPLO='L')
    eigenvectors = torch.matmul(D_over_sqrt, eigenvectors)
    return eigenvectors[:, 1:2].squeeze(1).cpu().numpy(), eigenvalues[1].cpu().numpy()


def get_salient_areas(second_smallest_vec):
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    return second_smallest_vec > avg


def segment_ids_to_mask(selected_ids, unique_segments):
    selected_map = torch.zeros_like(unique_segments)
    for s_id in selected_ids:
        segment_index = (unique_segments == s_id).nonzero(as_tuple=True)[0]
        selected_map[segment_index] = 1
    return selected_map.bool()


def ncut_iterative(sp_feats, unique_segments, segment_ids,
                   affinity_tau=0.6, max_instances=20, max_extent_ratio=0.6,
                   min_segment_size=4, eps=1e-5, connec_mask=None):
    """
    Run iterative NCut (from MyTrellis/playground/unscene3d.py).

    Returns:
        bipartitions: (K, N_sp) bool numpy array
        eigvalues: (K,) numpy array of second-smallest eigenvalues
    """
    bipartitions = []
    eigvalues = []
    foreground_segments = set()
    device = sp_feats.device

    if len(unique_segments) < 3:
        return np.ones(len(unique_segments), dtype=bool).reshape(1, -1), np.array([0.0])

    num_segments = len(unique_segments)
    for i in range(max_instances):
        if i == 0:
            painting = torch.zeros(num_segments, device=device)
        else:
            sp_feats, painting = get_masked_affinity_matrix(painting, sp_feats, current_mask)

        A, D = get_affinity_matrix(sp_feats, tau=affinity_tau, eps=eps, connec_mask=connec_mask)
        A[painting.cpu().bool()] = eps
        A[:, painting.cpu().bool()] = eps

        try:
            second_smallest_vec, eigenvalue = second_smallest_eigenvector(A, D)
        except Exception:
            break

        bipartition = get_salient_areas(second_smallest_vec)

        point_bipartition = bipartition[segment_ids.cpu()]
        is_fg_ratio_condition = point_bipartition.sum() / len(point_bipartition) > max_extent_ratio
        if is_fg_ratio_condition:
            bipartition = np.logical_not(bipartition)

        if bipartition.sum() < min_segment_size:
            current_mask = torch.from_numpy(bipartition).to(device)
            continue

        separated_seed_partition = set(unique_segments.cpu().numpy()[bipartition])
        separated_seed_partition_masked = separated_seed_partition - foreground_segments
        bipartitions.append(segment_ids_to_mask(separated_seed_partition_masked, unique_segments).cpu().numpy())
        eigvalues.append(eigenvalue)
        foreground_segments = foreground_segments.union(separated_seed_partition)

        current_mask = torch.from_numpy(bipartition).to(device)

    if len(bipartitions) == 0:
        return np.zeros((0, len(segment_ids)), dtype=bool), np.array([])
    return np.stack(bipartitions), np.stack(eigvalues)


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing (from MyTrellis/playground/ncut.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_sp_adjacency(sp_nbr, num_superpoints):
    """Build symmetric adjacency matrix from neighbor dict."""
    sp_adj = np.zeros((num_superpoints, num_superpoints), dtype=np.bool_)
    for sp_id, neighbors in sp_nbr.items():
        if sp_id >= num_superpoints or not neighbors:
            continue
        neighbors = np.asarray(neighbors, dtype=np.int64)
        sp_adj[sp_id, neighbors[neighbors < num_superpoints]] = True
    sp_adj |= sp_adj.T
    np.fill_diagonal(sp_adj, False)
    return sp_adj


def split_mask_by_adjacency(mask, adj_matrix, score=None):
    """Split masks whose superpoints form disconnected components under adj_matrix."""
    new_mask_list = []
    new_mask_score = []
    K, N = mask.shape
    mask = mask.copy()

    for i in range(K):
        valid_indices = np.where(mask[i] == 1)[0]
        if len(valid_indices) == 0:
            continue
        sub_adj = adj_matrix[valid_indices][:, valid_indices].astype(np.int32)
        np.fill_diagonal(sub_adj, 1)
        n_regions, region_labels = connected_components(
            csgraph=csr_matrix(sub_adj), directed=False, return_labels=True)
        if n_regions <= 1:
            continue
        for j in range(1, n_regions):
            rel_j_indices = np.where(region_labels == j)[0]
            if len(rel_j_indices) == 0:
                continue
            target_indices = valid_indices[rel_j_indices]
            new_mask = np.zeros((1, N), dtype=mask.dtype)
            new_mask[0, target_indices] = 1
            new_mask_list.append(new_mask)
            if score is not None:
                new_mask_score.append(score[i])
            mask[i, target_indices] = 0

    non_empty_mask = mask[np.sum(mask, axis=1) > 0]
    if score is not None:
        non_empty_mask_score = score[np.sum(mask, axis=1) > 0]

    if len(new_mask_list) > 0:
        new_masks = np.concatenate(new_mask_list, axis=0)
        final_mask = np.concatenate([new_masks, non_empty_mask], axis=0)
        if score is not None:
            new_mask_score = np.stack(new_mask_score)
            final_mask_score = np.concatenate([new_mask_score, non_empty_mask_score], 0)
    else:
        final_mask = non_empty_mask
        if score is not None:
            final_mask_score = non_empty_mask_score

    if score is not None:
        return final_mask, final_mask_score
    else:
        return final_mask


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def write_ply(path, coords, colors):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(coords)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(coords, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate pseudo masks via iterative NCut")
    p.add_argument("--data-root", default='../data/scannet/processed_aligns')
    p.add_argument("--superpoint-dir", default='../data/scannet/superpoints')
    p.add_argument("--feature-dir", default='../data/scannet/dinov2b14_spfeats')
    # p.add_argument("--feature-dir", default='/home/zihui/SSD/FoundObj/data/scannet/utonia/stage234')
    p.add_argument("--neighbor-dir", default='../data/scannet/superpoint_neighbors',
                   help="Dir with precomputed neighbor files (section 2 = dist<=0.05 adjacency).")
    p.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "pseudo_mask"))
    p.add_argument("--affinity-tau", type=float, default=0.8)
    p.add_argument("--max-instances", type=int, default=20)
    p.add_argument("--max-extent-ratio", type=float, default=0.6)
    p.add_argument("--min-segment-size", type=int, default=4)
    p.add_argument("--min-points", type=int, default=100, help="Min points per mask")
    p.add_argument("--max-points", type=int, default=8000, help="Max points per mask")
    p.add_argument("--max-bbox-xy", type=float, default=4.0)
    p.add_argument("--min-bbox-xy", type=float, default=0.15)
    p.add_argument("--max-bbox-z", type=float, default=2.5)
    p.add_argument("--visualize", default=True, help="Write instance PLY for each scene")
    p.add_argument("--evaluate", default=True, help="Run AP evaluation against GT")
    p.add_argument("--gt-dir", default="/home/zihui/SSD/FoundObj/data/scannet/processed_aligns/instance_gt/train")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    scenes = list_train_scenes(args.data_root)#[:100]
    print(f"Processing {len(scenes)} scenes → {args.output_dir}")

    for scene_idx, scan_id in enumerate(scenes):
        scene_name = f"scene{scan_id}" if not scan_id.startswith("scene") else scan_id
        output_path = os.path.join(args.output_dir, f"{scene_name}.pth")
        if os.path.exists(output_path):
            continue

        t0 = time.time()

        # Load data
        sp_path = os.path.join(args.superpoint_dir, f"{scene_name}_sp.npy")
        feat_path = os.path.join(args.feature_dir, f"{scene_name}.pth")
        points_path = os.path.join(args.data_root, "train", f"{scan_id}.npy")

        if not os.path.exists(feat_path):
            print(f"[{scene_idx+1}/{len(scenes)}] skip {scene_name}: no features")
            continue

        # Load points and voxelize
        points = np.load(points_path)
        pc = points[:, :3].astype(np.float32)
        pc = pc - pc.min(0)
        grids, unique_map, inverse_map = voxelize(pc)
        pc_voxel = pc[unique_map]

        # Load superpoints and remap after voxelization
        sp_ids = np.load(sp_path).astype(np.int64)
        sp_idx_voxel = sp_ids[unique_map]
        sp_idx_voxel_copy = -np.ones_like(sp_idx_voxel)
        valid_sp_idx = sp_idx_voxel[sp_idx_voxel != -1]
        unique_vals = np.unique(valid_sp_idx)
        unique_vals.sort()
        sp_idx_voxel_copy[sp_idx_voxel != -1] = np.searchsorted(unique_vals, valid_sp_idx)
        sp_idx_voxel = sp_idx_voxel_copy

        # Load SP features (already SP-level)
        sp_feats = torch.load(feat_path, weights_only=False).float()
        sp_feats = sp_feats[unique_vals]

        # Load the precomputed dist <= 0.05 adjacency (section 2 of the neighbor file,
        # produced by preprocessing/get_neighbors.py on the SAME superpoints as here).
        # It is indexed by raw superpoint id; slice to the voxel-surviving sps so the
        # rows/cols match the remapped sp ids. Identical to the on-the-fly
        # query_pairs(r=0.05) build, but read instead of recomputed.
        nbr_path = os.path.join(args.neighbor_dir, f"{scan_id}.npz")
        dist_full = load_dist_adjacency(nbr_path)
        if dist_full is None:
            raise RuntimeError(f"{nbr_path} is missing the dist adjacency section; "
                               f"re-run preprocessing/get_neighbors.py to regenerate it.")
        sp_adj = dist_full[unique_vals][:, unique_vals].copy()

        sp_idx_voxel_t = torch.from_numpy(sp_idx_voxel).long()

        # Filter: height/size based heuristic for possible object superpoints
        valid_mask = sp_idx_voxel_t != -1
        sp_size = scatter_sum(torch.ones_like(sp_idx_voxel_t[valid_mask]).cuda(),
                              sp_idx_voxel_t[valid_mask].cuda(), dim=0)
        sp_height = scatter_mean(torch.from_numpy(pc_voxel[valid_mask.numpy()][:, -1]).cuda(),
                                 sp_idx_voxel_t[valid_mask].cuda(), dim=0)
        nonobj_sp_mask = torch.logical_or(torch.logical_and(torch.logical_or(sp_height < 0.3, sp_height > 1.8),
                                                            sp_size > 300), sp_size < 10)
        possible_obj_sp_idx = torch.where(~nonobj_sp_mask.cpu())[0]

        # Subset to possible object superpoints
        sp_mask = torch.isin(sp_idx_voxel_t, possible_obj_sp_idx)
        sub_sp_idx = sp_idx_voxel_t[sp_mask]
        sub_unique = torch.unique(sub_sp_idx)
        sub_adj = sp_adj[sub_unique.numpy()][:, sub_unique.numpy()]
        sub_sp_feats = sp_feats[sub_unique].cuda()

        # Remap sub_sp_idx to contiguous 0..N
        sub_sp_idx_np = sub_sp_idx.cpu().numpy()
        sub_unique_np = sub_unique.cpu().numpy()
        sub_sp_idx_remapped = np.searchsorted(sub_unique_np, sub_sp_idx_np)
        sub_sp_idx_remapped = torch.from_numpy(sub_sp_idx_remapped).long()

        sub_sp_feats = F.normalize(sub_sp_feats, p=2, dim=1)
        unique_segments = torch.arange(len(sub_unique)).long()

        # Connectivity mask from neighbor adjacency.
        # Keep the diagonal True (self-loops) to match the reference, which uses
        # connec_mask = dist_matrix <= 0.05 (dist[i,i] = 0). Without self-loops the
        # affinity self-similarity A[i,i] is zeroed, changing the degree / normalized
        # Laplacian and thus every eigenvector. sub_adj itself (used for splitting,
        # which sets the diagonal internally) is left untouched.
        connec_mask = sub_adj.copy()
        np.fill_diagonal(connec_mask, True)

        # Run NCut
        mask, eigvalues = ncut_iterative(sub_sp_feats, unique_segments, sub_sp_idx_remapped,
            affinity_tau=args.affinity_tau, max_instances=args.max_instances,
            max_extent_ratio=args.max_extent_ratio, min_segment_size=args.min_segment_size,
            connec_mask=connec_mask)

        if mask.shape[0] == 0:
            print(f"[{scene_idx+1}/{len(scenes)}] {scene_name}: 0 instances, skip")
            continue

        # Split disconnected regions by adjacency
        mask, cutscore = split_mask_by_adjacency(mask, sub_adj, score=-eigvalues)

        # Map SP-level masks to voxel-level points
        point_mask = mask.T[sub_sp_idx_remapped.numpy()]  # (N_sub_points, K)

        # Filter masks by size and bounding box
        valid_mask_ids = []
        for i in range(point_mask.shape[1]):
            n_pts = point_mask[:, i].sum()
            if n_pts < args.min_points or n_pts > args.max_points:
                continue
            mask_pc = pc_voxel[sp_mask.numpy()][point_mask[:, i] == 1]
            mask_bbox = mask_pc.max(0) - mask_pc.min(0)
            if (mask_bbox[0:2].max() > args.min_bbox_xy and
                mask_bbox[0:2].max() < args.max_bbox_xy and
                mask_bbox[2] < args.max_bbox_z):
                valid_mask_ids.append(i)

        if len(valid_mask_ids) == 0:
            print(f"[{scene_idx+1}/{len(scenes)}] {scene_name}: 0 valid masks after filtering")
            continue

        point_mask = point_mask[:, valid_mask_ids]
        cutscore = cutscore[valid_mask_ids]

        # Map back to full voxelized point cloud
        full_mask = np.zeros((pc_voxel.shape[0], point_mask.shape[1]), dtype=bool)
        full_mask[sp_mask.numpy()] = point_mask

        # Map to original resolution via inverse_map
        full_mask_orig = full_mask[inverse_map]

        # Save
        torch.save(
            {'mask': torch.from_numpy(full_mask_orig).bool(),
             'score': torch.from_numpy(cutscore).float()},
            output_path)

        elapsed = time.time() - t0
        print(f"[{scene_idx+1}/{len(scenes)}] {scene_name}: "
              f"{point_mask.shape[1]} instances, {elapsed:.1f}s")

        # Optional visualization
        if args.visualize and point_mask.shape[1] > 0:
            np.random.seed(0)
            instance_colors = np.zeros((len(pc), 3))
            for i in reversed(range(full_mask_orig.shape[1])):
                instance_colors[full_mask_orig[:, i]] = np.random.rand(3)
            ply_path = os.path.join(args.output_dir, f"{scene_name}_instances.ply")
            write_ply(ply_path, pc - pc.min(0) + points[:, :3].min(0), (instance_colors * 255).astype(np.uint8))

    # Evaluation
    if args.evaluate:
        from benchmark.evaluate_semantic_instance import evaluate
        all_preds, all_gt, all_sem_preds = {}, {}, {}
        for scan_id in scenes:
            scene_name = f"scene{scan_id}" if not scan_id.startswith("scene") else scan_id
            output_path = os.path.join(args.output_dir, f"{scene_name}.pth")
            gt_file = os.path.join(args.gt_dir, f"{scene_name}.txt")
            if not os.path.exists(output_path) or not os.path.exists(gt_file):
                continue

            data = torch.load(output_path, weights_only=False)
            pred_mask = data['mask'].numpy().astype(np.float32)  # (N, K)
            pred_scores = pred_mask.sum(0)

            all_preds[scan_id] = {
                "pred_masks": pred_mask,
                "pred_scores": pred_scores,
                "pred_classes": np.ones(pred_mask.shape[1]),
            }
            all_gt[scan_id] = gt_file

            # Semantic eval: assign GT label to each predicted mask
            points = np.load(os.path.join(args.data_root, "train", f"{scan_id}.npy"))
            semantic = torch.from_numpy(points[:, 10].astype(np.int64))
            pred_sem = []
            for mask_id in range(pred_mask.shape[1]):
                pts_in_mask = np.where(pred_mask[:, mask_id] == 1)[0]
                if len(pts_in_mask) > 0:
                    sem = torch.mode(semantic[pts_in_mask]).values.item()
                else:
                    sem = 0
                pred_sem.append(sem)
            all_sem_preds[scan_id] = {
                "pred_masks": pred_mask,
                "pred_scores": pred_scores,
                "pred_classes": np.array(pred_sem),
            }

        print(f"\n{'='*60}")
        print("Class-agnostic evaluation:")
        print(f"{'='*60}")
        evaluate(False, all_preds, all_gt)
        print(f"\n{'='*60}")
        print("Semantic evaluation:")
        print(f"{'='*60}")
        evaluate(True, all_sem_preds, all_gt)


if __name__ == "__main__":
    main()



###
# ####################################################################################################
# what           :      AP  AP_50%  AP_25% |      RC  RC_50%  RC_25% |      PR  PR_50%  PR_25%
# ####################################################################################################
# class_agnostic :   0.067   0.183   0.440 |   0.180   0.307   0.474 |   0.222   0.415   0.661
# ----------------------------------------------------------------------------------------------------
# average        :   0.067   0.183   0.440 |   0.180   0.307   0.474 |   0.222   0.415   0.661
#
#
# ============================================================
# Semantic evaluation:
# ============================================================
# evaluating 1201 scans...
# scans processed: 1201
#
# ####################################################################################################
# what           :      AP  AP_50%  AP_25% |      RC  RC_50%  RC_25% |      PR  PR_50%  PR_25%
# ####################################################################################################
# cabinet        :   0.032   0.106   0.359 |   0.105   0.212   0.394 |   0.132   0.276   0.564
# bed            :   0.036   0.134   0.386 |   0.076   0.201   0.419 |   0.049   0.131   0.307
# chair          :   0.137   0.289   0.511 |   0.277   0.420   0.539 |   0.383   0.588   0.782
# sofa           :   0.121   0.301   0.486 |   0.198   0.357   0.495 |   0.199   0.362   0.549
# table          :   0.095   0.262   0.500 |   0.208   0.371   0.516 |   0.240   0.436   0.666
# door           :   0.058   0.168   0.375 |   0.135   0.248   0.386 |   0.246   0.472   0.739
# window         :   0.054   0.151   0.370 |   0.133   0.257   0.405 |   0.148   0.289   0.484
# bookshelf      :   0.037   0.126   0.367 |   0.102   0.211   0.398 |   0.072   0.153   0.316
# picture        :   0.071   0.183   0.343 |   0.132   0.261   0.364 |   0.273   0.545   0.769
# counter        :   0.012   0.039   0.352 |   0.048   0.112   0.358 |   0.060   0.145   0.513
# desk           :   0.008   0.039   0.303 |   0.030   0.096   0.336 |   0.028   0.095   0.401
# curtain        :   0.048   0.127   0.326 |   0.117   0.209   0.370 |   0.130   0.237   0.464
# refrigerator   :   0.121   0.279   0.491 |   0.229   0.374   0.497 |   0.299   0.507   0.710
# shower curtain :   0.246   0.508   0.656 |   0.405   0.569   0.664 |   0.525   0.742   0.865
# toilet         :   0.194   0.464   0.646 |   0.343   0.542   0.647 |   0.420   0.712   0.818
# sink           :   0.057   0.145   0.268 |   0.112   0.197   0.274 |   0.341   0.611   0.877
# bathtub        :   0.021   0.125   0.289 |   0.039   0.168   0.301 |   0.044   0.192   0.366
# otherfurniture :   0.063   0.161   0.388 |   0.166   0.285   0.433 |   0.267   0.468   0.727
# ----------------------------------------------------------------------------------------------------
# average        :   0.078   0.200   0.412 |   0.159   0.283   0.433 |   0.214   0.387   0.607



##U stage3
# ####################################################################################################
# what           :      AP  AP_50%  AP_25% |      RC  RC_50%  RC_25% |      PR  PR_50%  PR_25%
# ####################################################################################################
# class_agnostic :   0.070   0.198   0.487 |   0.196   0.343   0.538 |   0.217   0.413   0.671
# ----------------------------------------------------------------------------------------------------
# average        :   0.070   0.198   0.487 |   0.196   0.343   0.538 |   0.217   0.413   0.671
# average        :   0.083   0.216   0.474 |   0.173   0.314   0.505 |   0.219   0.399   0.634

## stage34
# ####################################################################################################
# what           :      AP  AP_50%  AP_25% |      RC  RC_50%  RC_25% |      PR  PR_50%  PR_25%
# ####################################################################################################
# class_agnostic :   0.071   0.183   0.426 |   0.186   0.300   0.452 |   0.271   0.481   0.737
# ----------------------------------------------------------------------------------------------------
# average        :   0.071   0.183   0.426 |   0.186   0.300   0.452 |   0.271   0.481   0.737
# ----------------------------------------------------------------------------------------------------
# average        :   0.069   0.171   0.375 |   0.146   0.251   0.394 |   0.235   0.412   0.660


## satge234
# ####################################################################################################
# what           :      AP  AP_50%  AP_25% |      RC  RC_50%  RC_25% |      PR  PR_50%  PR_25%
# ####################################################################################################
# class_agnostic :   0.071   0.183   0.425 |   0.185   0.299   0.452 |   0.271   0.481   0.737
# ----------------------------------------------------------------------------------------------------
# average        :   0.071   0.183   0.425 |   0.185   0.299   0.452 |   0.271   0.481   0.737
# ----------------------------------------------------------------------------------------------------
# average        :   0.070   0.174   0.379 |   0.147   0.254   0.399 |   0.234   0.413   0.660