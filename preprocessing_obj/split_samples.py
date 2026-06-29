"""Step 5: turn each labeled scene point cloud into five training samples.

Per scene, emit five training sample types:
  object   - the clean main object points
  box      - a (slightly expanded) bounding-box crop around the object
  fragment - a local spherical fragment of the object
  crop     - a non-object crop (from this scene, or an online ScanNet crop)
  multi    - the whole multi-object scene
"""
import os
import sys
import glob
import shutil
import json
import argparse
import numpy as np
import trimesh
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
from lib.helper_ply import read_ply, write_ply
from preprocessing_obj.dataset_utils import discover_datasets
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss.point_mesh_distance import point_face_distance
from torch_scatter import scatter_mean

OVERLAP_THRESHOLD = 0.2    # object/non-object voxel-overlap threshold for the samples


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


def voxelize(coords, voxel_size=0.02):
    scale = 1 / voxel_size
    coords = coords - coords.min(0)
    coords = np.floor(coords * scale)
    _, unique_map, inverse_map = np.unique(coords, return_index=True, return_inverse=True, axis=0)
    return unique_map, inverse_map


def sp_idx_smooth(sp_idx):
    result = -np.ones_like(sp_idx)
    valid = sp_idx != -1
    unique_vals = np.unique(sp_idx[valid])
    unique_vals.sort()
    result[valid] = np.searchsorted(unique_vals, sp_idx[valid])
    return result


def load_other_objects(mesh_dir, pc_dir, item):
    other_full = {}
    other_meshes = {}
    center_loc = os.path.join(mesh_dir, item, "center_loc.json")
    with open(center_loc, 'r') as f:
        center_data = json.load(f)

    for key in center_data.keys():
        if not key.startswith("other_"):
            continue
        full_pc = trimesh.load(os.path.join(pc_dir, item, key + ".ply")).vertices.astype(np.float32)
        other_full[key] = full_pc
        m = trimesh.load(os.path.join(mesh_dir, item, key + ".ply"), force='mesh')
        verts = torch.from_numpy(m.vertices.astype(np.float32)).cuda()
        faces = torch.from_numpy(m.faces.astype(np.int64)).cuda()
        other_meshes[key] = Meshes(verts=[verts], faces=[faces])

    return other_full, other_meshes


def check_other_overlap(crop_pc, crop_idx, other_meshes, other_full, overlap_threshold):
    remove_indices = []
    for key, full_pts in other_full.items():
        df = point2mesh_distance(other_meshes[key], crop_pc)
        hit = np.where(df < 0.01)[0]
        if hit.size == 0:
            continue
        full_normalized, center, scale = normalize_pts(full_pts)
        full_vox = points_to_voxel_grid(full_normalized)
        hit_normalized = (crop_pc[hit] - center) / scale
        hit_vox = points_to_voxel_grid(hit_normalized)
        if calculate_overlap(hit_vox, full_vox) >= overlap_threshold:
            remove_indices.extend(crop_idx[hit])
    return remove_indices


def obj_box(scene_pc, bbox_bounds, mesh_dir, item, pc_dir, full_pc, overlap_threshold):
    min_bound, max_bound = np.array(bbox_bounds[0]), np.array(bbox_bounds[1])
    box_center = (max_bound + min_bound) / 2
    other_full, other_meshes = load_other_objects(mesh_dir, pc_dir, item)
    full_normalized, center, scale = normalize_pts(full_pc)
    obj_voxel = points_to_voxel_grid(full_normalized)

    for tries in range(6):
        scaled_size = (max_bound - min_bound) * np.random.uniform(1, 1.2)
        expand_mode = np.random.choice([1, 2, 3])
        if expand_mode == 1:
            scaled_min = box_center - scaled_size / 2
            scaled_max = box_center + scaled_size / 2
        elif expand_mode == 2:
            scaled_min = min_bound
            scaled_max = max_bound + scaled_size * 0.5
        else:
            scaled_max = max_bound
            scaled_min = min_bound - scaled_size * 0.5

        mask = np.all((scene_pc >= scaled_min) & (scene_pc <= scaled_max), axis=1)
        box_pc = scene_pc[mask]
        if box_pc.shape[0] == 0:
            continue

        box_normalized = (box_pc - center) / scale
        if calculate_overlap(points_to_voxel_grid(box_normalized), obj_voxel) < overlap_threshold:
            continue

        if not other_full:
            return box_pc, mask

        box_idx = np.where(mask)[0]
        remove_indices = check_other_overlap(box_pc, box_idx, other_meshes, other_full, overlap_threshold)
        if not remove_indices:
            return box_pc, mask
        if tries == 5:
            mask[remove_indices] = False
            return scene_pc[mask], mask

    return None, None


def obj_frag(obj_point, full_point, overlap_threshold):
    full_normalized, center, scale = normalize_pts(full_point)
    full_voxel = points_to_voxel_grid(full_normalized)
    frag_point, frag_mask = None, None

    for count_try in range(20):
        scale_factor = np.random.uniform(0.4, 0.7)
        if count_try >= 2:
            scale_factor -= 0.2
        pt_center = obj_point[np.random.randint(0, len(obj_point))]
        if np.random.rand() < 0.5:
            dist = np.linalg.norm(obj_point - pt_center, axis=1)
        else:
            dist = np.linalg.norm(obj_point[:, :2] - pt_center[:2], axis=1)
        radius = np.max(dist) * scale_factor
        mask = dist <= radius
        candidate = obj_point[mask]
        if candidate.shape[0] > 10:
            frag_normalized = (candidate - center) / scale
            overlap = calculate_overlap(points_to_voxel_grid(frag_normalized), full_voxel)
            if overlap < overlap_threshold:
                frag_point, frag_mask = candidate, mask

    return frag_point, frag_mask


def random_crop(pc, scale_range=(0.3, 1)):
    if len(pc) == 0:
        return pc, np.zeros(len(pc), dtype=bool)
    center = pc[np.random.randint(0, len(pc))]
    if np.random.rand() < 0.5:
        distances = np.linalg.norm(pc - center, axis=1)
    else:
        distances = np.linalg.norm(pc[:, :2] - center[:2], axis=1)
    crop_radius = np.max(distances) * np.random.uniform(*scale_range)
    mask = distances <= crop_radius
    return pc[mask], mask


def simulate_nonobj(scene_pc, full_obj_pc, overlap_threshold, mesh_dir, item, pc_dir):
    other_full, other_meshes = load_other_objects(mesh_dir, pc_dir, item)
    full_normalized, center, scale = normalize_pts(full_obj_pc)
    obj_voxel = points_to_voxel_grid(full_normalized)

    for attempt in range(20):
        scene_crop, crop_mask = random_crop(scene_pc, scale_range=(0.3, 0.8))
        if scene_crop.size == 0:
            continue
        crop_idx = np.where(crop_mask)[0]
        scene_normalized = (scene_crop - center) / scale
        if calculate_overlap(points_to_voxel_grid(scene_normalized), obj_voxel) >= overlap_threshold:
            continue
        remove_indices = check_other_overlap(scene_crop, crop_idx, other_meshes, other_full, overlap_threshold)
        if not remove_indices:
            return scene_crop
        if attempt >= 10:
            crop_mask[remove_indices] = False
            return scene_pc[crop_mask]
    return None


def online_crop_scannet_nonobj(scene_list, overlap_threshold):
    for _ in range(20):
        scene_name = scene_list[np.random.randint(0, len(scene_list))]
        points = np.load(scene_name)
        pc, normal, sp_idx = points[:, :3], points[:, 6:9], points[:, 9]
        semantic, instance = points[:, 10].squeeze(), points[:, 11].squeeze()

        unique_map, _ = voxelize(pc.astype(np.float32))
        pc, normal, semantic, instance, sp_idx = (
            pc[unique_map], normal[unique_map], semantic[unique_map],
            instance[unique_map], sp_idx[unique_map])
        sp_idx = torch.from_numpy(sp_idx_smooth(sp_idx)).cuda().long()
        semantic_t = torch.from_numpy(semantic)

        gtmask = []
        for inst_id in np.unique(instance):
            if inst_id == -1:
                continue
            obj_mask = torch.from_numpy(inst_id == instance)
            label = torch.mode(semantic_t[torch.where(obj_mask)[0]]).values
            if label in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]:
                gtmask.append(obj_mask[None, ...])
        if not gtmask:
            continue
        gtmask = torch.cat(gtmask)

        center = pc[np.random.choice(len(pc), 1)]
        radius = np.random.uniform(0.25, 1.5)
        nonobjmask = torch.from_numpy(((pc - center) ** 2).sum(-1) < radius)
        tmp = scatter_mean(nonobjmask.float().cuda(), sp_idx, dim=0) > 0.5
        nonobjmask = tmp[sp_idx].cpu()
        if nonobjmask.sum() < 50:
            continue

        max_overlap = 0
        for obj_mask in gtmask:
            obj_pts = pc[obj_mask.numpy()]
            if len(obj_pts) < 10:
                continue
            obj_norm, c, s = normalize_pts(obj_pts)
            nonobj_norm = (pc[nonobjmask] - c) / s
            overlap = calculate_overlap(points_to_voxel_grid(nonobj_norm), points_to_voxel_grid(obj_norm))
            max_overlap = max(max_overlap, overlap)

        if max_overlap < overlap_threshold:
            return pc[nonobjmask], normal[nonobjmask]
    return None, None


def process_dataset(pc_dir, scene_pc_dir, mesh_dir, out_dir, scannet_train_list, overlap_threshold):
    if not os.path.isdir(scene_pc_dir):
        print(f"skip: {scene_pc_dir} not found")
        return
    os.makedirs(out_dir, exist_ok=True)
    items = sorted(os.listdir(scene_pc_dir))

    for i, item in enumerate(items, 1):
        item = item.split(".ply")[0]
        out_ply = os.path.join(out_dir, item, "object.ply")
        if os.path.exists(out_ply) and os.path.exists(os.path.join(out_dir, item, "center_loc.json")):
            print(f"skip {out_ply}")
            continue

        data = read_ply(os.path.join(scene_pc_dir, item + ".ply"))
        scene_pc = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
        mask = data['values']
        scene_mesh = trimesh.load_mesh(os.path.join(mesh_dir, item, "full_scene.ply"))
        obj_mask = (mask == 1).astype(bool)
        obj_mesh = trimesh.load(os.path.join(mesh_dir, item, 'object.ply'), force='mesh')

        obj_pc = scene_pc[obj_mask]
        if len(obj_pc) <= 100 or (obj_pc.max(0) - obj_pc.min(0)).min() <= 0.1:
            print(f"skip {item}: too few obj points")
            continue

        os.makedirs(os.path.join(out_dir, item), exist_ok=True)

        # 1. Object points
        write_ply(out_ply, [obj_pc], ['x', 'y', 'z'])

        full_pc = trimesh.load(os.path.join(pc_dir, item, "object.ply")).vertices.astype(np.float32)

        # 2. Bounding box crop
        inbox_points, inbox_mask = obj_box(scene_pc, obj_mesh.bounding_box.bounds,
                                           mesh_dir, item, pc_dir, full_pc, overlap_threshold)
        if inbox_points is not None and len(inbox_points) > 100:
            write_ply(os.path.join(out_dir, item, "box.ply"),
                      [inbox_points, mask[np.where(inbox_mask)[0]]], ['x', 'y', 'z', 'values'])

        # 3. Object fragment
        frag_point, _ = obj_frag(obj_pc, full_pc, overlap_threshold)
        if frag_point is not None and len(frag_point) > 100:
            write_ply(os.path.join(out_dir, item, "fragment.ply"), [frag_point], ['x', 'y', 'z'])

        # 4. Non-object crop
        crop = None
        if np.random.rand() < 0.5:
            crop = simulate_nonobj(scene_pc, full_pc, overlap_threshold, mesh_dir, item, pc_dir)
        elif scannet_train_list:
            try:
                crop, _ = online_crop_scannet_nonobj(scannet_train_list, overlap_threshold)
            except Exception as e:
                print(e)
        if crop is not None and len(crop) > 100:
            write_ply(os.path.join(out_dir, item, "crop.ply"), [crop.astype(np.float32)], ['x', 'y', 'z'])

        # 5. Multi-object (whole scene)
        shutil.copy(os.path.join(scene_pc_dir, item + ".ply"), os.path.join(out_dir, item, "multi.ply"))
        shutil.copy(os.path.join(mesh_dir, item, 'center_loc.json'), os.path.join(out_dir, item, 'center_loc.json'))

        print(f"completed {i}/{len(items)}")


def main():
    parser = argparse.ArgumentParser(description="Generate training samples (object, box, fragment, crop, multi)")
    parser.add_argument("--data_root", default="data/objects")
    parser.add_argument("--scannet_dir", default="data/scannet/processed_aligns/train",
                        help="required ScanNet processed train dir for non-object crops")
    args = parser.parse_args()

    scannet_train_list = sorted(glob.glob(os.path.join(args.scannet_dir, '*.npy')))
    if not scannet_train_list:
        raise FileNotFoundError(f"No ScanNet .npy files found in {args.scannet_dir}")
    for name in discover_datasets(args.data_root):
        print(f"\n=== dataset: {name} ===")
        ds_dir = os.path.join(args.data_root, name)
        process_dataset(
            os.path.join(ds_dir, "syn_depth"),
            os.path.join(ds_dir, "syn_scene_pc"),
            os.path.join(ds_dir, "syn_mesh"),
            os.path.join(ds_dir, "syn_traindata"),
            scannet_train_list, OVERLAP_THRESHOLD)


if __name__ == '__main__':
    main()
