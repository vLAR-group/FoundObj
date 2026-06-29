import argparse
import glob
import os
import time

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def group_point_indices(superpoint_indices: np.ndarray):
    valid_indices = np.flatnonzero(superpoint_indices != -1)
    valid_sp_ids = superpoint_indices[valid_indices]
    if len(valid_sp_ids) == 0:
        return []

    order = np.argsort(valid_sp_ids, kind="stable")
    sorted_indices = valid_indices[order]
    sorted_sp_ids = valid_sp_ids[order]
    sp_ids, starts = np.unique(sorted_sp_ids, return_index=True)
    ends = np.r_[starts[1:], len(sorted_indices)]
    return [(int(sp_id), sorted_indices[start:end]) for sp_id, start, end in zip(sp_ids, starts, ends)]


def find_superpoint_neighbors(point_cloud: np.ndarray, superpoint_indices: np.ndarray,
                              radius: float = 0.1, k: int = 30,
                              min_neighbor_points: int = 30, workers: int = -1):
    start_time = time.time()
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    superpoint_indices = np.asarray(superpoint_indices, dtype=np.int64).reshape(-1)
    if len(point_cloud) != len(superpoint_indices):
        raise ValueError(
            f"Point/superpoint length mismatch: {len(point_cloud)} points vs "
            f"{len(superpoint_indices)} superpoint ids")

    grouped_points = group_point_indices(superpoint_indices)
    all_sp_ids = [sp_id for sp_id, _ in grouped_points]
    print(f"Found {len(all_sp_ids)} superpoints")

    kd_tree = cKDTree(point_cloud)
    if radius is None:
        distances, _ = kd_tree.query(point_cloud, k=2, workers=workers)
        radius = 8 * np.mean(distances[:, 1])
        print(f"Auto radius: {radius:.6f}")

    k = min(k, len(point_cloud))
    _, nn_indices = kd_tree.query(point_cloud, k=k, workers=workers)
    if k == 1:
        nn_indices = nn_indices[:, None]
    nn_sp_ids = superpoint_indices[nn_indices]
    boundary_mask = (superpoint_indices != -1) & np.any(
        (nn_sp_ids != -1) & (nn_sp_ids != superpoint_indices[:, None]), axis=1)

    boundary_chunks = []
    source_chunks = []
    for sp_id, point_indices in grouped_points:
        boundary_indices = point_indices[boundary_mask[point_indices]]
        if len(boundary_indices) == 0:
            boundary_indices = point_indices
        boundary_chunks.append(boundary_indices)
        source_chunks.append(np.full(len(boundary_indices), sp_id, dtype=np.int64))

    neighbors = {sp_id: [] for sp_id in all_sp_ids}
    if boundary_chunks:
        boundary_indices = np.concatenate(boundary_chunks)
        source_sp_ids = np.concatenate(source_chunks)
        boundary_tree = cKDTree(point_cloud[boundary_indices])
        neighbor_matrix = boundary_tree.sparse_distance_matrix(kd_tree, radius, output_type="coo_matrix")

        source_sp_ids = source_sp_ids[neighbor_matrix.row]
        target_sp_ids = superpoint_indices[neighbor_matrix.col]
        valid = (target_sp_ids != -1) & (target_sp_ids != source_sp_ids)
        source_sp_ids = source_sp_ids[valid]
        target_sp_ids = target_sp_ids[valid]

        if len(source_sp_ids) > 0:
            label_stride = max(all_sp_ids) + 1
            pair_codes, counts = np.unique(
                source_sp_ids * label_stride + target_sp_ids, return_counts=True)
            keep = counts >= min_neighbor_points
            source_ids = pair_codes[keep] // label_stride
            target_ids = pair_codes[keep] % label_stride
            counts = counts[keep]

            order = np.lexsort((target_ids, -counts, source_ids))
            for source_sp_id, target_sp_id in zip(source_ids[order], target_ids[order]):
                neighbors[int(source_sp_id)].append(int(target_sp_id))

    print(f"Elapsed: {time.time() - start_time:.2f}s")
    return neighbors


def compute_sp_dist_adjacency(point_cloud: np.ndarray, superpoint_indices: np.ndarray,
                              radius: float = 0.05):
    """Distance-based superpoint adjacency as a dense bool matrix.

    `adj[i, j] = True` iff superpoints i and j have a pair of points within
    `radius` (i.e. single-linkage min distance <= radius). Computed in ONE
    cKDTree.query_pairs pass over the full-resolution cloud, which reproduces a
    precomputed distance matrix thresholded at `radius` (e.g. RLTrellis
    `dis_matrixes_initseg_unscene3d` with dist <= 0.05) EXACTLY: "exists a point
    pair within r" is identical to "min_point_distance <= r".

    The matrix is indexed by raw superpoint id (size (max_sp_id + 1)^2), so it can
    be sliced by the surviving ids after voxelization just like the neighbor dict.
    The diagonal is True (dist[i, i] = 0 <= radius), matching the reference.
    """
    start_time = time.time()
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    superpoint_indices = np.asarray(superpoint_indices, dtype=np.int64).reshape(-1)

    valid = superpoint_indices != -1
    valid_pc = point_cloud[valid]
    valid_sp = superpoint_indices[valid]
    num_sp = int(valid_sp.max()) + 1 if len(valid_sp) else 0

    adj = np.zeros((num_sp, num_sp), dtype=np.bool_)
    if num_sp == 0:
        return adj
    np.fill_diagonal(adj, True)

    tree = cKDTree(valid_pc)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) > 0:
        sp_a = valid_sp[pairs[:, 0]]
        sp_b = valid_sp[pairs[:, 1]]
        cross = sp_a != sp_b
        adj[sp_a[cross], sp_b[cross]] = True
        adj[sp_b[cross], sp_a[cross]] = True

    print(f"Dist adjacency elapsed: {time.time() - start_time:.2f}s")
    return adj


def _write_neighbor_section(f, superpoint_ids, neighbors) -> None:
    f.write(len(superpoint_ids).to_bytes(4, byteorder="little"))
    for sp_id in superpoint_ids:
        sp_neighbors = neighbors[sp_id]
        f.write(int(sp_id).to_bytes(4, byteorder="little"))
        f.write(len(sp_neighbors).to_bytes(4, byteorder="little"))
        for neighbor in sp_neighbors:
            f.write(int(neighbor).to_bytes(4, byteorder="little"))


def _read_neighbor_section(f):
    head = f.read(4)
    if len(head) < 4:
        return None
    num_sp = int.from_bytes(head, byteorder="little")
    neighbor_dict = {}
    for _ in range(num_sp):
        sp_id = int.from_bytes(f.read(4), byteorder="little")
        num_neighbors = int.from_bytes(f.read(4), byteorder="little")
        neighbor_dict[sp_id] = [
            int.from_bytes(f.read(4), byteorder="little")
            for _ in range(num_neighbors)
        ]
    return neighbor_dict


def save_neighbors(path: str, superpoint_ids, neighbors, dist_adj=None) -> None:
    """Write the contact-based neighbor dict (section 1) and, if provided, the
    distance-based bool adjacency matrix (section 2) into a single file.

    Section 2 layout: N (uint32) followed by N*N bool bytes (row-major).
    The legacy single-section loader (load_superpoint_neighbors) reads only
    section 1 and ignores the trailing bytes, so it stays backward compatible.
    """
    with open(path, "wb") as f:
        _write_neighbor_section(f, superpoint_ids, neighbors)
        if dist_adj is not None:
            dist_adj = np.ascontiguousarray(dist_adj, dtype=np.bool_)
            f.write(int(dist_adj.shape[0]).to_bytes(4, byteorder="little"))
            f.write(dist_adj.tobytes())
    print(f"Saved neighbor file: {path}")


def load_superpoint_neighbors(load_path: str):
    """Load the contact-based neighbor dict (section 1). Backward compatible."""
    with open(load_path, "rb") as f:
        return _read_neighbor_section(f)


def load_dist_adjacency(load_path: str):
    """Load the distance-based bool adjacency matrix (section 2), or None if the
    file predates it."""
    with open(load_path, "rb") as f:
        _read_neighbor_section(f)  # skip section 1
        head = f.read(4)
        if len(head) < 4:
            return None
        num_sp = int.from_bytes(head, byteorder="little")
        buf = f.read(num_sp * num_sp)
        if len(buf) < num_sp * num_sp:
            return None
        return np.frombuffer(buf, dtype=np.bool_).reshape(num_sp, num_sp).copy()


def load_neighbors_and_dist(load_path: str):
    """Load both sections as (contact_neighbors, dist_adjacency). dist_adjacency
    is None for legacy files that only contain section 1."""
    with open(load_path, "rb") as f:
        neighbors = _read_neighbor_section(f)
        head = f.read(4)
        if len(head) < 4:
            return neighbors, None
        num_sp = int.from_bytes(head, byteorder="little")
        buf = f.read(num_sp * num_sp)
        dist = None
        if len(buf) == num_sp * num_sp:
            dist = np.frombuffer(buf, dtype=np.bool_).reshape(num_sp, num_sp).copy()
    return neighbors, dist


def _file_has_dist(path: str) -> bool:
    """True if `path` already contains the distance-based section 2."""
    try:
        with open(path, "rb") as f:
            if _read_neighbor_section(f) is None:
                return False
            return len(f.read(4)) == 4
    except Exception:
        return False


def scene_name_from_npy(npy_path: str) -> str:
    scan_id = os.path.splitext(os.path.basename(npy_path))[0]
    return scan_id if scan_id.startswith("scene") else f"scene{scan_id}"


def process_scene(args: argparse.Namespace, npy_path: str) -> None:
    scan_id = os.path.splitext(os.path.basename(npy_path))[0]
    scene_name = scene_name_from_npy(npy_path)
    output_path = os.path.join(args.output_dir, f"{scan_id}.npz")
    sp_path = os.path.join(args.superpoint_dir, f"{scene_name}_sp.npy")

    if os.path.exists(output_path) and _file_has_dist(output_path):
        print(f"skip {scan_id}")
        return
    if not os.path.exists(sp_path):
        raise FileNotFoundError(f"Superpoint file not found: {sp_path}")

    point_cloud = np.load(npy_path)[:, :3].astype(np.float32)
    superpoint_indices = np.load(sp_path).astype(np.int64).reshape(-1)
    neighbors = find_superpoint_neighbors(
        point_cloud, superpoint_indices, radius=args.radius, k=args.k,
        min_neighbor_points=args.min_neighbor_points, workers=args.workers)
    dist_adj = compute_sp_dist_adjacency(
        point_cloud, superpoint_indices, radius=args.dist_radius)

    os.makedirs(args.output_dir, exist_ok=True)
    save_neighbors(output_path, sorted(neighbors), neighbors, dist_adj=dist_adj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data", "scannet", "processed_aligns"))
    parser.add_argument("--superpoint-dir", default=os.path.join(REPO_ROOT, "data", "scannet", "superpoints"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "data", "scannet", "superpoint_neighbors"))
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--dist-radius", type=float, default=0.05,
                        help="Radius for the distance-based bool adjacency (dist <= r).")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--min-neighbor-points", type=int, default=30)
    parser.add_argument("--workers", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        npy_paths = sorted(glob.glob(os.path.join(args.data_root, split, "*.npy")))
        print(f"{split}: {len(npy_paths)} scenes")
        for npy_path in npy_paths:
            process_scene(args, npy_path)


if __name__ == "__main__":
    main()
