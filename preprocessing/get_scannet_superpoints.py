import argparse
import glob
import os
import sys
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CPP_UTILS = os.path.join(SCRIPT_DIR, "cpp_utils")
for cpp_path in [CPP_UTILS] + glob.glob(os.path.join(CPP_UTILS, "build", "lib.*")):
    sys.path.insert(0, cpp_path)

FELZENSZWALB_CPP = None


def load_felzenszwalb_cpp():
    global FELZENSZWALB_CPP
    if FELZENSZWALB_CPP is not None:
        return FELZENSZWALB_CPP

    try:
        import felzenszwalb_cpp
    except ImportError as exc:
        raise ImportError(
            "Cannot import felzenszwalb_cpp. Compile it first:\n"
            f"  cd {CPP_UTILS}\n"
            "  python setup.py build_ext --inplace") from exc

    FELZENSZWALB_CPP = felzenszwalb_cpp
    return FELZENSZWALB_CPP


def voxelize(coords: np.ndarray, voxel_size: float):
    """Map points to a regular voxel grid, matching how the reference superpoints
    (RLTrellis/outputs) were voxelized.

    Uses the absolute grid ``floor(coords / voxel_size)`` WITHOUT subtracting the min,
    computed in float64. Subtracting the min (as generate_ncut_masks.py does) shifts the
    grid origin, and float32 flips voxel-boundary assignments — either breaks bit-exact
    reproduction of the reference.

    Returns the unique voxel coords, the representative-point index per voxel
    (unique_map), and the per-point voxel index (inverse_map).
    """
    grid = np.floor(coords.astype(np.float64) / voxel_size)
    grid, unique_map, inverse_map = np.unique(grid, return_index=True, return_inverse=True, axis=0)
    return grid, unique_map, inverse_map


def load_axis_alignment(scene_dir: str, scene_name: str) -> np.ndarray:
    info_file = os.path.join(scene_dir, f"{scene_name}.txt")
    if not os.path.exists(info_file):
        return np.eye(4, dtype=np.float32)

    with open(info_file, "r") as f:
        for line in f:
            if line.startswith("axisAlignment"):
                values = np.fromstring(line.split(" = ", 1)[1], sep=" ")
                return values.reshape(4, 4).astype(np.float32)

    return np.eye(4, dtype=np.float32)


def load_aligned_mesh(scans_root: str, scene_name: str) -> o3d.geometry.TriangleMesh:
    scene_dir = os.path.join(scans_root, scene_name)
    mesh_path = os.path.join(scene_dir, f"{scene_name}_vh_clean_2.ply")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {mesh_path}")
    mesh.transform(load_axis_alignment(scene_dir, scene_name))
    return mesh


def segment_mesh(mesh: o3d.geometry.TriangleMesh, threshold: float, min_vertices: int) -> np.ndarray:
    felzenszwalb_cpp = load_felzenszwalb_cpp()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if len(colors) == 0:
        colors = np.zeros((len(vertices), 3), dtype=np.float32)

    segment_ids, _ = felzenszwalb_cpp.segment_mesh(vertices, faces, colors, threshold, min_vertices)
    return segment_ids.astype(np.int32)


def save_superpoint_ply(output_path: str, points: np.ndarray, segment_ids: np.ndarray) -> None:
    colors = np.stack((segment_ids * 217 % 256, segment_ids * 217 % 311, segment_ids * 217 % 541), axis=1) % 256
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(output_path, pcd)


def process_scene(args: argparse.Namespace, npy_path: str) -> None:
    scan_id = os.path.splitext(os.path.basename(npy_path))[0]
    scene_name = scan_id if scan_id.startswith("scene") else f"scene{scan_id}"
    sp_path = os.path.join(args.output_dir, f"{scene_name}_sp.npy")
    ply_path = os.path.join(args.output_dir, f"{scene_name}_sp.ply")

    if os.path.exists(sp_path):
        print(f"skip {scene_name}")
        return

    mesh = load_aligned_mesh(args.scans_root, scene_name)
    mesh_segment_ids = segment_mesh(mesh, args.threshold, args.min_vertices)

    points = np.load(npy_path)[:, :3].astype(np.float32)
    nearest_vertex = KDTree(np.asarray(mesh.vertices, dtype=np.float32)).query(points, k=1)[1].reshape(-1)
    segment_ids = mesh_segment_ids[nearest_vertex]

    # Collapse superpoints onto the 0.02 voxel grid: every point in a voxel inherits the
    # superpoint of that voxel's representative point. This reproduces the reference
    # superpoints (RLTrellis/outputs), which were voxelized with floor(coords/0.02) before
    # use. Output is still kept at full resolution (one label per point).
    _, unique_map, inverse_map = voxelize(points, args.voxel_size)
    segment_ids = segment_ids[unique_map][inverse_map]

    _, segment_ids = np.unique(segment_ids, return_inverse=True)
    segment_ids = segment_ids.astype(np.int32)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(sp_path, segment_ids)
    if args.save_ply:
        save_superpoint_ply(ply_path, points, segment_ids)
    print(f"{scene_name}: {len(points)} points, {len(np.unique(segment_ids))} superpoints")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans-root", default='/home/zihui/HDD/ScanNetv2/scans')
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data", "scannet", "processed_aligns"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "data", "scannet", "superpoints"))
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--min-vertices", type=int, default=50)
    parser.add_argument("--voxel-size", type=float, default=0.02,
                        help="Voxel grid size for collapsing superpoints (match generate_ncut_masks.py)")
    parser.add_argument("--save-ply", action="store_true", default=False)
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
