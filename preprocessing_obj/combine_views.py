"""Step 4: overlap-guided multi-view selection and labeled scene point cloud merge.

For each scene, depth views are merged until the visible part of the main object
covers enough of its full surface (voxel overlap >= threshold). Every merged point
is then labeled by nearest-mesh distance: 0 = background, 1 = main object,
>1 = a sufficiently-visible other object.
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import trimesh
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
from lib.helper_ply import write_ply
from preprocessing_obj.dataset_utils import discover_datasets
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.structures import Meshes, Pointclouds
os.environ['PYOPENGL_PLATFORM'] = 'egl'

OVERLAP_THRESHOLD = 0.2    # min object visibility (voxel overlap) to accept a merge

colormap = np.array([
    [245, 130,  48], [  0, 130, 200], [ 60, 180,  75], [255, 225,  25], [145,  30, 180],
    [250, 190, 190], [230, 190, 255], [210, 245,  60], [240,  50, 230], [ 70, 240, 240]])


def point2mesh_distance(mesh, points, device="cuda"):
    pcls = Pointclouds(points=[torch.from_numpy(points.astype(np.float32)).to(device)])
    pts = pcls.points_packed()
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    tris = mesh.verts_packed()[mesh.faces_packed()]
    tris_first_idx = mesh.mesh_to_faces_packed_first_idx()
    df = point_face_distance(pts, points_first_idx, tris, tris_first_idx, max_points).sqrt()
    return df.squeeze(0).cpu().numpy()


def points_to_voxel_grid(points, resolution=32):
    if points is None or len(points) == 0:
        return set()
    voxel_coords = np.floor((points + 0.5) * resolution).astype(int)
    voxel_coords = np.clip(voxel_coords, 0, resolution - 1)
    return set(map(tuple, voxel_coords))


def calculate_overlap(voxels1, voxels2):
    intersection = voxels1.intersection(voxels2)
    return len(intersection) / len(voxels2) if len(voxels2) > 0 else 0.0


def normalize_pts(pts):
    if isinstance(pts, trimesh.Trimesh):
        coords = pts.vertices.astype(np.float32)
    elif isinstance(pts, np.ndarray):
        coords = pts.astype(np.float32)
    else:
        raise TypeError(f"Unsupported type {type(pts)}")
    min_coords, max_coords = coords.min(axis=0), coords.max(axis=0)
    center = (min_coords + max_coords) / 2
    scale = (max_coords - min_coords).max() + 1e-5
    return (coords - center) / scale, center, scale


def overlap_view_selection(all_pcs, obj_mesh, surf_pc, floor_mesh, overlap_threshold=0.2):
    device = "cuda"
    verts = torch.from_numpy(obj_mesh.vertices.astype(np.float32)).to(device)
    faces = torch.from_numpy(obj_mesh.faces.astype(np.int64)).to(device)
    obj_mesh_pt3d = Meshes(verts=[verts], faces=[faces])

    obj_surf_normalized, center, scale = normalize_pts(surf_pc)
    obj_voxels = points_to_voxel_grid(obj_surf_normalized)

    all_combined = np.vstack(all_pcs)
    available_indices = list(range(len(all_pcs)))
    selected_indices = []

    while available_indices:
        candidate = np.random.choice(available_indices)
        selected_indices.append(candidate)
        available_indices.remove(candidate)

        current_combined = np.vstack([all_pcs[idx] for idx in selected_indices])
        df = point2mesh_distance(obj_mesh_pt3d, current_combined)
        floor_df = point2mesh_distance(floor_mesh, current_combined)
        current_obj = current_combined[(df < 0.01) & (floor_df >= 0.01)]
        current_obj_normalized = (current_obj - center) / scale
        current_voxels = points_to_voxel_grid(current_obj_normalized)
        overlap = calculate_overlap(current_voxels, obj_voxels)

        if overlap >= overlap_threshold:
            return all_combined, current_combined, df, obj_surf_normalized

    return None, None, None, None


def load_and_sample_views(dep_dir, mesh_dir, item, floor_mesh):
    mesh = trimesh.load(os.path.join(mesh_dir, 'object.ply'), force='mesh')
    surf_pc = trimesh.load(os.path.join(dep_dir, 'object.ply')).vertices.astype(np.float32)
    ply_files = sorted(glob.glob(os.path.join(dep_dir, 'dep*.ply')))
    all_pcs = []
    for ply_file in ply_files:
        vertices = trimesh.load(ply_file).vertices.astype(np.float32)
        if vertices is not None and len(vertices) > 0:
            all_pcs.append(vertices)
    return overlap_view_selection(all_pcs, mesh, surf_pc, floor_mesh)


def process_dataset(pc_dir, mesh_dir, out_dir, overlap_threshold):
    if not os.path.isdir(pc_dir):
        print(f"skip: {pc_dir} not found")
        return
    os.makedirs(out_dir, exist_ok=True)
    items = sorted(os.listdir(pc_dir))
    device = "cuda"

    for i, item in enumerate(items, 1):
        out_ply = os.path.join(out_dir, item + ".ply")
        if os.path.exists(out_ply):
            continue

        floor_mesh = trimesh.load(os.path.join(mesh_dir, item, 'floor.ply'), force='mesh')
        verts = torch.from_numpy(floor_mesh.vertices.astype(np.float32)).to(device)
        faces = torch.from_numpy(floor_mesh.faces.astype(np.int64)).to(device)
        floor_mesh_pt3d = Meshes(verts=[verts], faces=[faces])

        wall_mesh_pt3d = None
        if os.path.exists(os.path.join(mesh_dir, item, 'wall.ply')):
            wall_mesh = trimesh.load(os.path.join(mesh_dir, item, 'wall.ply'), force='mesh')
            verts = torch.from_numpy(wall_mesh.vertices.astype(np.float32)).to(device)
            faces = torch.from_numpy(wall_mesh.faces.astype(np.int64)).to(device)
            wall_mesh_pt3d = Meshes(verts=[verts], faces=[faces])

        all_pc, combined_pc, obj_df, _ = load_and_sample_views(
            os.path.join(pc_dir, item), os.path.join(mesh_dir, item), item, floor_mesh_pt3d)
        if combined_pc is None:
            print(f"skip {item}: cannot reach overlap threshold")
            continue

        floor_df = point2mesh_distance(floor_mesh_pt3d, combined_pc)
        dfs = [obj_df[:, None]]

        json_file = os.path.join(mesh_dir, item, "center_loc.json")
        with open(json_file, 'r') as f:
            data = json.load(f)

        for key in data.keys():
            if not key.startswith("other_"):
                continue
            other_mesh = trimesh.load(os.path.join(mesh_dir, item, key + '.ply'), force='mesh')
            verts = torch.from_numpy(other_mesh.vertices.astype(np.float32)).to(device)
            faces = torch.from_numpy(other_mesh.faces.astype(np.int64)).to(device)
            other_mesh_pt3d = Meshes(verts=[verts], faces=[faces])

            other_surf = trimesh.load(os.path.join(pc_dir, item, key + '.ply')).vertices.astype(np.float32)
            other_surf_normalized, center, scale = normalize_pts(other_surf)
            other_voxels = points_to_voxel_grid(other_surf_normalized)

            df = point2mesh_distance(other_mesh_pt3d, combined_pc)
            other_pts = combined_pc[(df < 0.01) & (floor_df >= 0.01)]
            other_pts_normalized = (other_pts - center) / scale
            other_vox = points_to_voxel_grid(other_pts_normalized)
            overlap = calculate_overlap(other_vox, other_voxels)

            if overlap >= overlap_threshold:
                dfs.append(df[:, None])
            else:
                dfs.append(np.ones((df.shape[0], 1), dtype=np.float32) * 10.0)

        dfs = np.concatenate(dfs, axis=1)
        final_mask = np.argmin(dfs, axis=1).astype(np.float32) + 1
        final_mask[dfs.min(axis=1) > 0.01] = 0
        final_mask[floor_df < 0.01] = 0
        if wall_mesh_pt3d is not None:
            wall_df = point2mesh_distance(wall_mesh_pt3d, combined_pc)
            final_mask[(wall_df < 0.01) & (obj_df < 0.01)] = 0

        color = np.zeros_like(combined_pc)
        for k in range(dfs.shape[1]):
            color[final_mask == k + 1] = colormap[k]

        write_ply(out_ply, [combined_pc.astype(np.float32), color.astype(np.uint8), final_mask],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'values'])
        print(f"completed {i}/{len(items)}")


def main():
    parser = argparse.ArgumentParser(description="Overlap-guided multi-view selection and scene point cloud merging")
    parser.add_argument("--data_root", default="data/objects")
    args = parser.parse_args()

    for name in discover_datasets(args.data_root):
        print(f"\n=== dataset: {name} ===")
        ds_dir = os.path.join(args.data_root, name)
        process_dataset(
            os.path.join(ds_dir, "syn_depth"),
            os.path.join(ds_dir, "syn_mesh"),
            os.path.join(ds_dir, "syn_scene_pc"),
            OVERLAP_THRESHOLD)


if __name__ == '__main__':
    main()
