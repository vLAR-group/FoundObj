import os

import numpy as np
import scipy
import scipy.interpolate
import torch
import yaml
from glob import glob
from torch.utils.data import Dataset

from lib.aug_tools import rota_coords, scale_coords
from preprocessing.scannet200_constants import VALID_CLASS_IDS_20

SUP_TRAIN_CLASS_IDS = [4, 5, 6, 7, 14, 24, 33, 34, 36, 39]


class VoxelizedDataset(Dataset):
    def __init__(self, mode, cfg, batch_size, num_workers=8):
        self.mode = mode
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_label = -1
        self.limit_numpoints = 1200000
        self.label_offset = 2
        self.filter_out_classes = [0, 1]

        self.data = []
        scene_list = sorted(glob(os.path.join(cfg.data_dir, mode, '*.npy')))
        for scene_name in scene_list:
            semantic = np.load(scene_name)[:, 10:11]
            if self.mode == 'train':
                if np.any(np.isin(semantic, SUP_TRAIN_CLASS_IDS)):
                    self.data.append(scene_name)
            else:
                self.data.append(scene_name)

        with open(os.path.join(cfg.data_dir, 'color_mean_std.yaml')) as f:
            mean_std = yaml.load(f, Loader=yaml.FullLoader)
        self.color_mean = np.array(mean_std['mean']) * 255
        self.color_inv_std = np.reciprocal(np.array(mean_std['std']) * 255)
        self.rota_coords = rota_coords(rotation_bound=((-0, 0), (-0, 0), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))

    def __len__(self):
        return len(self.data)

    def map_NYU_label(self, labels):
        labels[~np.isin(labels, list(VALID_CLASS_IDS_20))] = self.ignore_label
        for i, k in enumerate(VALID_CLASS_IDS_20):
            labels[labels == k] = i
        return labels

    def remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(VALID_CLASS_IDS_20):
            output_remapped[output == i] = k
        return output_remapped

    def elastic_distortion(self, pointcloud, granularity, magnitude):
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
        ax = [np.linspace(d_min, d_max, d) for d_min, d_max, d in
              zip(coords_min - granularity, coords_min + granularity * (noise_dim - 2), noise_dim)]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        pointcloud[:, :3] = coords + interp(coords) * magnitude
        return pointcloud

    def __getitem__(self, idx):
        path = self.data[idx]
        points = np.load(path)
        pc, color, sp_idx = points[:, :3], points[:, 3:6], points[:, 9]
        semantic, instance = points[:, 10:11], points[:, 11:12]
        sp_idx = sp_idx.astype(np.int32)

        scene_name = 'scene' + os.path.splitext(os.path.basename(path))[0]

        pc = pc - pc.min(0)
        raw_feature = np.concatenate([color, pc], 1)

        if self.mode == 'train':
            pc[:, 0:2] += (np.random.uniform(pc.min(0), pc.max(0)) / 2)[0:2][None]
            for i in (0, 1):
                if np.random.random() < 0.5:
                    pc[:, i] = pc[:, i].max() - pc[:, i]
            pc = self.scale_coords(pc)
            pc = self.rota_coords(pc)
            if np.random.random() < 0.9:
                for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                    pc = self.elastic_distortion(pc, granularity, magnitude)

        pc = pc.astype(np.float32)
        color = (color - self.color_mean) * self.color_inv_std

        semantic = self.map_NYU_label(semantic)
        for filter_class in self.filter_out_classes:
            semantic[semantic == filter_class] = self.ignore_label
        semantic[semantic != self.ignore_label] = np.clip(
            semantic[semantic != self.ignore_label] - self.label_offset, 0, None)

        feature = np.concatenate([color, pc], 1)
        coords, feature, semantic, instance, unique_map, inverse_map = self._voxelize(pc, feature, semantic, instance)
        feature = torch.from_numpy(feature)
        semantic = torch.from_numpy(semantic)
        instance = torch.from_numpy(instance)
        voxel_pc = torch.from_numpy(pc[unique_map])
        raw_feature = torch.from_numpy(raw_feature)

        sp_idx_voxel = sp_idx[unique_map]
        remapped = -np.ones_like(sp_idx_voxel)
        valid = sp_idx_voxel != -1
        if valid.any():
            unique_vals = np.unique(sp_idx_voxel[valid])
            unique_vals.sort()
            remapped[valid] = np.searchsorted(unique_vals, sp_idx_voxel[valid])
        sp_idx_voxel = remapped

        return (coords, feature, semantic.squeeze(), instance.squeeze(),
                inverse_map, unique_map, scene_name, voxel_pc, raw_feature,
                torch.from_numpy(sp_idx_voxel).long(), torch.from_numpy(sp_idx).long())

    def _voxelize(self, coords, feature, semantic, instance):
        scale = 1 / self.cfg.voxel_size
        coords = coords - coords.min(0)
        coords = np.floor(coords * scale)
        coords, unique_map, inverse_map = np.unique(coords, return_index=True, return_inverse=True, axis=0)
        return coords, feature[unique_map], semantic[unique_map], instance[unique_map], unique_map, inverse_map

    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=self._collate_fn, shuffle=shuffle, drop_last=True,
            worker_init_fn=lambda wid: np.random.seed(int.from_bytes(os.urandom(4), "big") + wid))

    def _collate_fn(self, batch):
        (coords, feature, semantic, instance, inverse_map, unique_map,
         scene_name, voxel_pc, raw_feature, voxel_sp, raw_sp) = list(zip(*batch))

        coords_batch, feature_batch = [], []
        voxel_pc_batch, raw_feature_batch = [], []
        voxel_sp_batch, raw_sp_batch = [], []
        semantic_batch, instance_batch = [], []
        target = []
        batch_num_points = 0

        for b in range(len(coords)):
            num_points = coords[b].shape[0]
            batch_num_points += num_points
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                break

            coords_batch.append(torch.cat((
                torch.full((num_points, 1), b, dtype=torch.int32),
                torch.from_numpy(coords[b]).int()), 1))
            feature_batch.append(feature[b])
            voxel_pc_batch.append(voxel_pc[b])
            raw_feature_batch.append(raw_feature[b])
            voxel_sp_batch.append(voxel_sp[b])
            raw_sp_batch.append(raw_sp[b])
            semantic_batch.append(semantic[b])
            instance_batch.append(instance[b] + semantic[b] * 1000)

            target.append(dict())

            valid_sp_mask = voxel_sp[b] != -1
            _, ret_index, _ = np.unique(
                voxel_sp[b][valid_sp_mask].numpy(), return_index=True, return_inverse=True)
            sp_instance_label = instance[b][ret_index]

            instance_ids = torch.unique(instance[b])
            instance_ids = instance_ids[instance_ids != -1]
            if instance_ids.numel() == 0:
                target[b]['labels'] = []
                target[b]['masks'] = torch.zeros_like(instance[b])[None, :]
                target[b]['segment_mask'] = []
            else:
                masks = instance_ids[:, None] == instance[b][None, :]
                labels = torch.stack([torch.mode(semantic[b][m]).values for m in masks]).long()
                sp_masks = instance_ids[:, None] == sp_instance_label[None, :]
                keep = labels >= 0
                if keep.any():
                    target[b]['labels'] = labels[keep]
                    target[b]['masks'] = masks[keep]
                    target[b]['segment_mask'] = sp_masks[keep]
                else:
                    target[b]['labels'] = []
                    target[b]['masks'] = torch.zeros_like(instance[b])[None, :]
                    target[b]['segment_mask'] = []

        coords_batch = torch.cat(coords_batch, 0)
        feature_batch = torch.cat(feature_batch, 0).float()
        actual_bs = len(voxel_pc_batch)
        return (coords_batch, feature_batch, target, scene_name[:actual_bs],
                semantic_batch, instance_batch, inverse_map[:actual_bs], unique_map[:actual_bs],
                voxel_pc_batch, raw_feature_batch, voxel_sp_batch, raw_sp_batch)
