"""Step 2: render each scene mesh from several random views into depth point clouds."""
import os
import sys
import torch
import argparse
import numpy as np
import trimesh
from pytorch3d.structures import Meshes, join_meshes_as_batch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
from preprocessing_obj.depth_render import DepthRender
from preprocessing_obj.dataset_utils import discover_datasets
from lib.helper_ply import write_ply

NUM_VIEWS = 12    # random depth views rendered per scene


def render_scene_views(mesh_path: str, save_root: str, depth_renderer: DepthRender):
    os.makedirs(save_root, exist_ok=True)
    mesh = trimesh.load(mesh_path, force='mesh')

    device = depth_renderer.device
    verts = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
    faces = torch.from_numpy(mesh.faces.astype(np.int64)).to(device)
    single_mesh = Meshes(verts=[verts], faces=[faces])
    mesh_batch = join_meshes_as_batch([single_mesh.clone() for _ in range(depth_renderer.render_num)])

    _, _, _, _, coords_obj_list = depth_renderer.render(mesh_batch)

    for idx, coords_obj in enumerate(coords_obj_list):
        if coords_obj.shape[0] > 10000:
            choice = np.random.choice(coords_obj.shape[0], 10000, replace=False)
            coords_obj = coords_obj[choice]
        write_ply(os.path.join(save_root, f"dep_pcl_{idx}.ply"), [coords_obj], ['x', 'y', 'z'])


def process_dataset(mesh_dir, out_root, num_views):
    if not os.path.isdir(mesh_dir):
        print(f"skip: {mesh_dir} not found")
        return
    os.makedirs(out_root, exist_ok=True)
    valid_folders = sorted([
        f for f in os.listdir(mesh_dir)
        if os.path.exists(os.path.join(mesh_dir, f, 'full_scene.ply'))
    ])
    for i, model_id in enumerate(valid_folders, 1):
        if os.path.exists(os.path.join(out_root, model_id, "dep_pcl_0.ply")):
            continue
        radius = 2.0
        azim = np.random.uniform(0, 180, num_views)
        elev = np.random.uniform(-30, 30, num_views)
        depth_renderer = DepthRender(np.full(num_views, radius), elev, azim, device="cuda")
        render_scene_views(os.path.join(mesh_dir, model_id, "full_scene.ply"),
                           os.path.join(out_root, model_id), depth_renderer)
        print(f"completed {i}/{len(valid_folders)}")


def main():
    parser = argparse.ArgumentParser(description="Multi-view depth rendering of constructed scenes")
    parser.add_argument("--data_root", default="data/objects")
    args = parser.parse_args()

    for name in discover_datasets(args.data_root):
        print(f"\n=== dataset: {name} ===")
        process_dataset(
            os.path.join(args.data_root, name, "syn_mesh"),
            os.path.join(args.data_root, name, "syn_depth"),
            NUM_VIEWS)


if __name__ == "__main__":
    main()
