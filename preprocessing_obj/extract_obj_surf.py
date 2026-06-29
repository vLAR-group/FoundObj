"""Step 3: render each object mesh from 20 even views to get a dense surface point
cloud (used to estimate occlusion / overlap in step 4)."""
import os
import sys
import math
import torch
import argparse
import numpy as np
import trimesh
from pytorch3d.structures import Meshes, join_meshes_as_batch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
from preprocessing_obj.depth_render import DepthRender
from preprocessing_obj.dataset_utils import discover_datasets
from lib.helper_ply import write_ply


def extract_surface(mesh_path: str, depth_renderer: DepthRender):
    mesh = trimesh.load(mesh_path, force='mesh')
    bbox = mesh.bounding_box.bounds
    center = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / scale)

    device = depth_renderer.device
    verts = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
    faces = torch.from_numpy(mesh.faces.astype(np.int64)).to(device)
    single_mesh = Meshes(verts=[verts], faces=[faces])
    mesh_batch = join_meshes_as_batch([single_mesh.clone() for _ in range(depth_renderer.render_num)])

    _, _, _, _, coords_obj_list = depth_renderer.render(mesh_batch)

    coords_obj = np.concatenate(coords_obj_list, axis=0)
    coords_obj = coords_obj * scale + center
    if coords_obj.shape[0] > 50000:
        choice = np.random.choice(coords_obj.shape[0], size=50000, replace=False)
        coords_obj = coords_obj[choice]
    return coords_obj


def fibonacci_sphere(count=20):
    increment = math.pi * (3 - math.sqrt(5))
    azim, elev = [], []
    for i in range(count):
        theta = math.asin(-1 + 2 * i / (count - 1))
        phi = ((i + 1) * increment) % (2 * math.pi)
        elev.append(phi / math.pi * 180)
        azim.append(theta / math.pi * 180)
    return elev, azim


def process_dataset(mesh_dir, out_root):
    if not os.path.isdir(mesh_dir):
        print(f"skip: {mesh_dir} not found")
        return
    os.makedirs(out_root, exist_ok=True)
    valid_folders = sorted([
        f for f in os.listdir(mesh_dir)
        if os.path.exists(os.path.join(mesh_dir, f, 'full_scene.ply'))
    ])
    for i, model_id in enumerate(valid_folders, 1):
        object_files = [f for f in os.listdir(os.path.join(mesh_dir, model_id)) if 'object' in f.lower()]
        for object_file in object_files:
            out_file = os.path.join(out_root, model_id, object_file)
            if os.path.exists(out_file):
                continue
            count = 20
            elev, azim = fibonacci_sphere(count)
            depth_renderer = DepthRender(np.full(count, 2.0), elev, azim, device="cuda")
            pc = extract_surface(os.path.join(mesh_dir, model_id, object_file), depth_renderer)
            os.makedirs(os.path.join(out_root, model_id), exist_ok=True)
            write_ply(out_file, [pc], ['x', 'y', 'z'])
        print(f"completed {i}/{len(valid_folders)}")


def main():
    parser = argparse.ArgumentParser(description="Extract dense surface point clouds from object meshes")
    parser.add_argument("--data_root", default="data/objects")
    args = parser.parse_args()

    for name in discover_datasets(args.data_root):
        print(f"\n=== dataset: {name} ===")
        process_dataset(
            os.path.join(args.data_root, name, "syn_mesh"),
            os.path.join(args.data_root, name, "syn_depth"))


if __name__ == "__main__":
    main()
