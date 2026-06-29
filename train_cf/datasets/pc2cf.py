import os
import numpy as np
import torch, json
from torch.utils.data import Dataset
import random, scipy
from scipy.spatial.transform import Rotation
from lib.helper_ply import read_ply


class PC2vector(Dataset):
    def __init__(self, roots: str):
        self.roots = [root for root in roots.split(',') if root]
        self.instances = []
        for root in self.roots:
            instance_dir = os.path.join(root, 'syn_traindata')
            if not os.path.isdir(instance_dir):
                raise FileNotFoundError(f"No syn_traindata directory found under {root}")
            for f in sorted(os.listdir(instance_dir)):
                instance_path = os.path.join(instance_dir, f)
                if os.path.isdir(instance_path):
                    self.instances.append(instance_path)
        if not self.instances:
            raise RuntimeError(f"No CenterField training samples found in: {roots}")

    def __len__(self):
        return len(self.instances)

    def elastic_distortion(self, pointcloud, granularity=(0.2, 0.4), magnitude=(0.8, 1.6)):
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords = pointcloud[:, :3]
        coords_min = coords.min(0)

        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

        ax = [np.linspace(d_min, d_max, d)
              for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity * (noise_dim - 2), noise_dim)]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        pointcloud[:, :3] = coords + interp(coords) * magnitude
        return pointcloud

    @staticmethod
    def random_rota():
        random_R_z = np.random.uniform(-1 * np.pi, 1 * np.pi)
        random_R_x = np.random.uniform(-np.pi / 18, np.pi / 18)
        random_R_y = np.random.uniform(-np.pi / 18, np.pi / 18)
        random_R = np.array([random_R_x, random_R_y, random_R_z]).astype(np.float32)
        return Rotation.from_euler('xyz', random_R).as_matrix().astype(np.float32)

    @staticmethod
    def normalize_pts(pts):
        min_coords, max_coords = pts.min(axis=0), pts.max(axis=0)
        center = (min_coords + max_coords) / 2
        scale = (max_coords - min_coords).max() + 1e-5
        return (pts - center) / scale, center, scale

    def prepare_center_field(self, plyfile, center_dict):
        data = read_ply(plyfile)
        if np.vstack((data['x'], data['y'], data['z'])).T.shape[0] == 0:
            raise ValueError(f"No points found in {plyfile}")
        if plyfile.endswith('object.ply'):
            pts = np.vstack((data['x'], data['y'], data['z'])).T
            center = np.array(center_dict['object'])
            cf = pts - center
            return pts, cf, np.ones(pts.shape[0])
        elif plyfile.endswith('box.ply'):
            pts, mask = np.vstack((data['x'], data['y'], data['z'])).T, data['values']
            center = np.array(center_dict['object'])
            cf = pts - center
            cf[mask == 0] = cf[mask == 0] * 0
            return pts, cf, mask
        elif plyfile.endswith('crop.ply') or plyfile.endswith('fragment.ply'):
            pts = np.vstack((data['x'], data['y'], data['z'])).T
            return pts, np.zeros_like(pts), np.zeros(pts.shape[0])
        elif plyfile.endswith('multi.ply'):
            pts, mask = np.vstack((data['x'], data['y'], data['z'])).T, data['values']
            centers = pts.copy()
            for i in np.unique(mask):
                if i > 0:
                    centers[mask == i] = np.array(center_dict[list(center_dict.keys())[int(i) - 1]])
            cf = pts - centers
            cf[mask == 0] = cf[mask == 0] * 0
            return pts, cf, mask
        else:
            raise NotImplementedError(f"Unsupported ply file type: {plyfile}")

    def get_instance(self, instance):
        ply_files = [f for f in os.listdir(instance) if
                     f.lower().endswith('.ply') and not f.lower().endswith('box2.ply')]
        assert len(ply_files) > 0, f"No PLY files found in {instance}"
        file = random.choice(ply_files)
        ply_file = os.path.join(instance, file)

        json_file = os.path.join(instance, "center_loc.json")
        with open(json_file, 'r') as f:
            center_dict = json.load(f)

        pts, _, mask = self.prepare_center_field(ply_file, center_dict)

        random_R = self.random_rota()
        pts = np.dot(pts, random_R.T)

        pts, center, scale = self.normalize_pts(pts)
        pts += 0.005 * np.random.randn(*pts.shape).astype(np.float32)

        if np.random.random() < 0.8:
            for granularity, magnitude in ((0.2, 0.4), (0.4, 0.8)):
                pts = self.elastic_distortion(pts, granularity, magnitude)

        pts = np.clip(pts, -0.5, 0.5)
        instance_mask = np.asarray(mask)
        fg_mask = np.ascontiguousarray(instance_mask != 0)
        if not ply_file.endswith('multi.ply'):
            center2 = np.array(center_dict['object'])
            center2 = np.dot(center2, random_R.T)
            center2 = (center2 - center) / scale

        else:
            center2 = pts.copy()
            for i in np.unique(instance_mask):
                if i > 0:
                    center_i = np.array(center_dict[list(center_dict.keys())[int(i) - 1]])
                    center_i = np.dot(center_i, random_R.T)
                    center_i = (center_i - center) / scale
                    center2[instance_mask == i] = center_i
        cf = pts - center2
        cf[~fg_mask] *= 0
        mask = fg_mask
        index = np.arange(pts.shape[0])
        index = np.random.choice(index, 5000)
        pts = pts[index]
        cf = cf[index]
        mask = mask[index]
        return torch.from_numpy(pts).float()[None], torch.from_numpy(cf).float()[None], torch.from_numpy(mask).float()[None]

    def __getitem__(self, index):
        instance = self.instances[index]
        return self.get_instance(instance)

    def collate_fn(self, batch):
        pts, cf, mask = list(zip(*batch))
        pts_lenseq = []
        for batch_id, _ in enumerate(pts):
            num_points = pts[batch_id].shape[1]
            pts_lenseq.append(num_points)
        return torch.cat(pts), torch.cat(cf), pts_lenseq, torch.cat(mask)
