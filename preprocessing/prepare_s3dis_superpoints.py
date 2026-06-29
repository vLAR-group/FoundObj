"""
Generate superpoints for S3DIS using cut-pursuit graph partitioning.
Streamlined version — no MinkowskiEngine dependency, numpy voxelization.
Same algorithm and output as zzh4090 original.
"""
import os
import time
import argparse
from glob import glob
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
import libcp
import libply_c


def voxelize(pc, voxel_size):
    """Numpy voxelization equivalent to ME.utils.sparse_quantize."""
    coords = np.floor(pc / voxel_size)
    _, unique_map, inverse_map = np.unique(
        coords, axis=0, return_index=True, return_inverse=True)
    return unique_map, inverse_map


def compute_graph(xyz, k_nn_adj, k_nn_geof):
    """KNN graph + neighbor indices for geof computation."""
    nn = NearestNeighbors(n_neighbors=k_nn_geof + 1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]

    target_fea = neighbors.flatten().astype(np.uint32)

    n_ver = xyz.shape[0]
    source = np.repeat(np.arange(n_ver), k_nn_adj).astype(np.uint32)
    target = neighbors[:, :k_nn_adj].flatten().astype(np.uint32)
    dists = distances[:, :k_nn_adj].flatten().astype(np.float32)

    return source, target, dists, target_fea


def write_ply(filename, points, colors):
    """Write a colored point cloud to PLY format."""
    n = points.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(filename, 'wb') as f:
        f.write(header.encode())
        data = np.empty(n, dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
                                  ('r','u1'),('g','u1'),('b','u1')])
        data['x'], data['y'], data['z'] = points[:,0], points[:,1], points[:,2]
        data['r'], data['g'], data['b'] = colors[:,0], colors[:,1], colors[:,2]
        data.tofile(f)


def generate_colormap(n):
    """Generate distinct colors for n superpoints."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 255, size=(n, 3)).astype(np.uint8)


def construct_superpoints(path, args):
    f = Path(path)
    scene_name = f"{f.parts[-2]}/{f.stem}"
    sp_file = os.path.join(args.sp_save_path, scene_name + '_superpoint.npy')

    if os.path.exists(sp_file):
        return

    points = np.load(path)
    pc = (points[:, :3] - points[:, :3].min(0)).astype(np.float32)
    rgb = points[:, 3:6].astype(np.float32) / 255

    t0 = time.time()

    unique_map, inverse_map = voxelize(pc, args.voxel_size)
    xyz = pc[unique_map]
    rgb_v = rgb[unique_map]

    source, target, dists, target_fea = compute_graph(xyz, args.k_nn_adj, args.k_nn_geof)

    geof = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype(np.float32)

    features = np.hstack((geof, rgb_v)).astype(np.float32)
    features[:, 3] *= 2.0  # boost verticality

    edge_weight = (1.0 / (1.0 + dists / dists.mean())).astype(np.float32)

    _, in_component = libcp.cutpursuit(features, source, target, edge_weight, args.reg_strength)

    # Relabel continuous
    _, sp_labels = np.unique(in_component, return_inverse=True)
    out_sp_labels = sp_labels[inverse_map].astype(np.int32)

    os.makedirs(os.path.dirname(sp_file), exist_ok=True)
    np.save(sp_file, out_sp_labels)

    n_sp = sp_labels.max() + 1

    # Save visualization
    if args.vis:
        vis_dir = os.path.join(args.sp_save_path, 'vis', f.parts[-2])
        os.makedirs(vis_dir, exist_ok=True)
        colormap = generate_colormap(n_sp)
        colors = colormap[out_sp_labels]
        write_ply(os.path.join(vis_dir, f.stem + '.ply'), pc, colors)

    print(f'{scene_name}: {time.time()-t0:.1f}s, {n_sp} superpoints')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="/home/zihui/SSD/FoundObj/data/s3dis/processed")
    parser.add_argument("--sp-save-path", type=str, default="/home/zihui/SSD/FoundObj/data/s3dis/SPG")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--reg-strength", type=float, default=0.05)
    parser.add_argument("--k-nn-geof", type=int, default=45)
    parser.add_argument("--k-nn-adj", type=int, default=10)
    parser.add_argument("--areas", default=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"])
    parser.add_argument("--vis", default=True)
    args = parser.parse_args()

    os.makedirs(args.sp_save_path, exist_ok=True)
    paths = []
    for area in args.areas:
        paths += sorted(glob(os.path.join(args.input_path, area, '*.npy')))

    print(f'{len(paths)} scenes to process')
    for p in paths:
        construct_superpoints(p, args)
    print('Done.')


if __name__ == "__main__":
    main()
