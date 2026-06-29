import os

import numpy as np
import torch
import yaml
from glob import glob
from torch.utils.data import Dataset


class S3DISDataset(Dataset):
    def __init__(self, mode, areas, cfg, batch_size=1, num_workers=4):
        self.path = cfg.data_dir
        self.mode = mode
        self.areas = areas
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxel_size = cfg.voxel_size
        self.limit_numpoints = 3000000

        self.data = []
        global_mean_std_path = os.path.join(self.path, 'color_mean_std.yaml')
        global_mean_std = self._load_yaml(global_mean_std_path) if os.path.exists(global_mean_std_path) else None
        for area in areas:
            scene_list = sorted(glob(os.path.join(self.path, area, '*.npy')))
            if global_mean_std:
                mean_std = global_mean_std
            else:
                mean_std = self._load_yaml(os.path.join(self.path, area + '_color_mean_std.yaml'))
            for scene_name in scene_list:
                self.data.append((scene_name, (mean_std['mean'], mean_std['std'], 255)))

    def __len__(self):
        return len(self.data)

    def _load_yaml(self, filepath):
        with open(filepath) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def __getitem__(self, idx):
        path, (mean, std, max_pixel_value) = self.data[idx]
        points = np.load(path)
        pc, color, sp_idx = points[:, :3], points[:, 3:6], points[:, 9]
        semantic, instance = points[:, 10:11], points[:, 11:12]

        pc = (pc - pc.min(0)).astype(np.float32)
        raw_rgbxyz = np.concatenate([color, pc], 1)

        mean = np.array(mean) * max_pixel_value
        std = np.array(std) * max_pixel_value
        color = (color - mean) * np.reciprocal(std)
        feature = np.concatenate([color, pc], 1)

        coords, feature, instance, unique_map, inverse_map = self._voxelize(pc, feature, instance)
        feature = torch.from_numpy(feature)
        semantic = torch.from_numpy(semantic)
        instance = torch.from_numpy(instance)
        pc_voxel = torch.from_numpy(pc[unique_map])
        pc_full = torch.from_numpy(pc)

        area_name = path.split('/')[-2]
        scene_name = area_name + '/' + os.path.splitext(os.path.basename(path))[0]

        if self.cfg.sp_dir is not None:
            mysp_path = os.path.join(self.cfg.sp_dir, scene_name + '_superpoint.npy')
            if os.path.exists(mysp_path):
                sp_idx = np.load(mysp_path)
                if len(sp_idx.shape) == 2:
                    sp_idx = sp_idx.squeeze(1)

        sp_idx_voxel = sp_idx[unique_map].astype(np.int64)
        valid = sp_idx_voxel != -1
        smoothed = -np.ones_like(sp_idx_voxel)
        if valid.any():
            unique_vals = np.unique(sp_idx_voxel[valid])
            unique_vals.sort()
            smoothed[valid] = np.searchsorted(unique_vals, sp_idx_voxel[valid])
        sp_idx_voxel = smoothed

        return (coords, feature, semantic.squeeze(), instance.squeeze(),
                inverse_map, unique_map, scene_name, pc_voxel, pc_full,
                torch.from_numpy(sp_idx_voxel).long(),
                torch.from_numpy(sp_idx.astype(np.int64)).long(),
                semantic.squeeze()[unique_map], raw_rgbxyz)

    def _voxelize(self, coords, feature, instance):
        scale = 1 / self.voxel_size
        coords = coords - coords.min(0)
        coords = np.floor(coords * scale)
        coords, unique_map, inverse_map = np.unique(coords, return_index=True, return_inverse=True, axis=0)
        return coords, feature[unique_map], instance[unique_map], unique_map, inverse_map

    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=self._collate_fn, shuffle=shuffle,
            worker_init_fn=lambda wid: np.random.seed(int.from_bytes(os.urandom(4), "big") + wid))

    def _collate_fn(self, batch):
        (coords, feature, full_semantic, instance, inverse_map, unique_map,
         scene_name, pc, pc_full, sp_idx, sp_idx_full, semantic, raw_rgbxyz) = list(zip(*batch))

        coords_batch, feature_batch = [], []
        pc_batch, pc_batch_full, sp_batch, sp_batch_full = [], [], [], []
        target = []
        semantic_batch = []
        batch_num_points = 0

        for batch_id in range(len(coords)):
            num_points = coords[batch_id].shape[0]
            batch_num_points += num_points
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                break

            coords_batch.append(torch.cat((
                torch.ones(num_points, 1).int() * batch_id,
                torch.from_numpy(coords[batch_id]).int()), 1))
            feature_batch.append(feature[batch_id])
            pc_batch.append(pc[batch_id])
            pc_batch_full.append(pc_full[batch_id])
            sp_batch.append(sp_idx[batch_id])
            sp_batch_full.append(sp_idx_full[batch_id])
            semantic_batch.append(semantic[batch_id])

            target.append(dict())
            valid_sp_mask = sp_idx[batch_id] != -1
            _, ret_index, _ = np.unique(
                sp_idx[batch_id][valid_sp_mask].numpy(), return_index=True, return_inverse=True)

            masks, labels = [], []
            for instance_id in torch.unique(instance[batch_id]):
                if instance_id == -1:
                    continue
                mask = (instance[batch_id] == instance_id).bool()
                label = torch.mode(semantic[batch_id][mask]).values
                if label == 8:
                    masks.append(mask.unsqueeze(0))
                    labels.append(torch.zeros_like(label.unsqueeze(0).long()))

            if masks:
                target[batch_id]['labels'] = torch.cat(labels)
                target[batch_id]['masks'] = torch.cat(masks, dim=0).squeeze(-1)
            else:
                target[batch_id]['labels'] = []
                target[batch_id]['masks'] = torch.zeros_like(instance[batch_id])[None, :]

        coords_batch = torch.cat(coords_batch, 0).float()
        feature_batch = torch.cat(feature_batch, 0).float()
        actual_bs = len(pc_batch)

        return (coords_batch, feature_batch, target, scene_name, semantic_batch,
                [instance[i] for i in range(actual_bs)],
                inverse_map, unique_map, pc_batch, pc_batch_full,
                sp_batch, sp_batch_full, full_semantic, raw_rgbxyz)
