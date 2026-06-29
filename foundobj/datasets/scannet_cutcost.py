import os
import warnings
import logging

import numpy as np
import scipy
import scipy.interpolate
import torch
import yaml
from glob import glob
from torch.utils.data import Dataset

from lib.aug_tools import rota_coords, scale_coords
from preprocessing.get_neighbors import load_neighbors_and_dist
from preprocessing.scannet200_constants import VALID_CLASS_IDS_20

TRAIN_OBJECT_CLASS_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


class VoxelizedDataset(Dataset):
    def __init__(self, mode, cfg, batch_size, RL=False):
        self.mode = mode
        self.cfg = cfg
        self.batch_size = batch_size
        self.ignore_label = -1
        self.limit_numpoints = 1200000
        self.label_offset = 2
        self.filter_out_classes = [0, 1]

        self.data, self.sp_nbr, self.dino_path, self.sp_path = [], [], [], []
        scene_list = sorted(glob(os.path.join(self.cfg.data_root, mode, '*.npy')))
        for scene_name in scene_list:
            scene_id = os.path.splitext(os.path.basename(scene_name))[0]
            if self.mode == 'train':
                semantic = np.load(scene_name)[:, 10:11]
                if not np.any(np.isin(semantic, TRAIN_OBJECT_CLASS_IDS)):
                    continue
                self.data.append(scene_name)
                self.sp_nbr.append(os.path.join(self.cfg.superpoint_neighbor_dir, scene_id + '.npz'))
                self.dino_path.append(os.path.join(self.cfg.dino_dir, 'scene' + scene_id + '.pth'))
                if self.cfg.superpoint_dir is not None:
                    self.sp_path.append(os.path.join(self.cfg.superpoint_dir, 'scene' + scene_id + '_sp.npy'))
            else:
                self.data.append(scene_name)

        if RL:
            self.data = self.data[:10]

        with open(os.path.join(self.cfg.data_root, 'color_mean_std.yaml')) as f:
            mean_std = yaml.load(f, Loader=yaml.FullLoader)
        self.color_mean = np.array(mean_std["mean"]) * 255
        self.color_inv_std = np.reciprocal(np.array(mean_std["std"]) * 255)
        self.rota_coords = rota_coords(rotation_bound=(None, None, (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))

        if self.mode == 'train':
            n_found = sum(
                os.path.exists(os.path.join(
                    self.cfg.pre_pseudo,
                    'scene' + os.path.splitext(os.path.basename(s))[0] + '.pth'))
                for s in self.data)
            logging.info(f'[pre_pseudo] found {n_found}/{len(self.data)} pre-pseudo masks in {self.cfg.pre_pseudo}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        points = np.load(self.data[idx])
        pc, color, sp_idx = points[:, :3], points[:, 3:6], points[:, 9]
        semantic, instance = points[:, 10:11], points[:, 11:12]

        if self.cfg.superpoint_dir is not None and self.mode == 'train':
            sp_idx = np.load(self.sp_path[idx])
        sp_idx = sp_idx.astype(np.int32)

        if self.mode == 'train':
            pc = self.augment_coords(pc)
        pc = pc.astype(np.float32)
        raw_feature = np.concatenate([color, pc], 1)
        pc = pc - pc.min(0)
        color = (color - self.color_mean) * self.color_inv_std
        semantic = self.map_NYU_label(semantic)
        semantic[np.isin(semantic, self.filter_out_classes)] = self.ignore_label
        semantic[semantic != self.ignore_label] = np.clip(semantic[semantic != self.ignore_label] - self.label_offset, 0, None)

        coords, feature, semantic, instance, unique_map, inverse_map = self.voxelize(
            pc, np.concatenate([color, pc], 1), semantic, instance)
        feature = torch.from_numpy(feature)
        semantic = torch.from_numpy(semantic)
        instance = torch.from_numpy(instance)
        voxel_pc = torch.from_numpy(pc[unique_map])
        raw_feature = torch.from_numpy(raw_feature)

        sp_idx_voxel, unique_vals = self.remake_sp_idx(sp_idx, unique_map)
        scene_name = 'scene' + os.path.splitext(os.path.basename(self.data[idx]))[0]

        if self.mode == 'train':
            train_extras = self._load_train_extras(idx, unique_map, unique_vals, sp_idx, color, pc, points)
        else:
            train_extras = dict(dino=None, sp_nbrs=None, dist_adj=None, exist_mask=None, prexist_mask=None,
                                coords_seg=None, feature_seg=None, inverse_map_seg=None, sp_idx_voxel_seg=None)

        return {"coords": coords, "feature": feature.float(), "semantic": semantic.squeeze(), "instance": instance.squeeze(),
            "inverse_map": inverse_map, "unique_map": unique_map, "scene_name": scene_name, "voxel_pc": voxel_pc,
            "raw_feature": raw_feature, "voxel_sp": torch.from_numpy(sp_idx_voxel).long(), **train_extras}

    def _load_train_extras(self, idx, unique_map, unique_vals, sp_idx, color, pc, points):
        sp_nbr, dist_full = load_neighbors_and_dist(self.sp_nbr[idx])
        if dist_full is None:
            raise RuntimeError(f"{self.sp_nbr[idx]} is missing the dist section; "f"re-run preprocessing/get_neighbors.py to regenerate it.")
        dino = torch.load(self.dino_path[idx], weights_only=False, map_location='cpu', mmap=True)
        dino_feats = dino[unique_vals].clone()
        sp_nbr = self.remap_nbr(sp_nbr, old_sp_ids=unique_vals, new_sp_ids=np.arange(len(unique_vals)))
        # dist_full is indexed by raw superpoint id; slice to the voxel-surviving sps.
        # Row/col order matches remake_sp_idx's searchsorted remapping, and the diagonal
        # is True (dist[i,i] = 0 <= radius), reproducing the RLTrellis dist <= 0.05 adjacency.
        dist_adj = dist_full[unique_vals][:, unique_vals]

        scene_name = 'scene' + os.path.splitext(os.path.basename(self.data[idx]))[0]
        exist_mask = self.load_exist_mask(scene_name, unique_map, len(unique_map))
        prexist_mask = self.load_prexist_mask(scene_name, unique_map)

        pc_seg = self.augment_coords(points[:, :3], elastic=True).astype(np.float32)
        pc_seg = pc_seg - pc_seg.min(0)
        coords_seg, feature_seg, unique_map_seg, inverse_map_seg = self.voxelize_seg(
            pc_seg, np.concatenate([color, pc_seg], 1))
        sp_idx_voxel_seg = self.remake_sp_idx(sp_idx, unique_map_seg)[0]

        return dict(dino=dino_feats, sp_nbrs=sp_nbr, dist_adj=dist_adj,
            exist_mask=exist_mask, prexist_mask=prexist_mask,
            coords_seg=coords_seg, feature_seg=torch.from_numpy(feature_seg),
            inverse_map_seg=inverse_map_seg,
            sp_idx_voxel_seg=torch.from_numpy(sp_idx_voxel_seg).long())

    # ──────────────────────────────────────────────────────────────────────
    # Label mapping
    # ──────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────
    # Augmentation
    # ──────────────────────────────────────────────────────────────────────

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

    def augment_coords(self, pc, elastic=False):
        pc[:, 0:2] += (np.random.uniform(pc.min(0), pc.max(0)) / 2)[0:2][None]
        for i in (0, 1):
            if np.random.random() < 0.5:
                pc[:, i] = pc[:, i].max() - pc[:, i]
        pc = self.scale_coords(pc)
        pc = self.rota_coords(pc)
        if elastic and np.random.random() < 0.9:
            for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                pc = self.elastic_distortion(pc, granularity, magnitude)
        return pc

    # ──────────────────────────────────────────────────────────────────────
    # Voxelization
    # ──────────────────────────────────────────────────────────────────────

    def voxelize(self, coords, feature, semantic, instance):
        coords, feature, unique_map, inverse_map = self.voxelize_seg(coords, feature)
        return coords, feature, semantic[unique_map], instance[unique_map], unique_map, inverse_map

    def voxelize_seg(self, coords, feature):
        coords = coords - coords.min(0)
        coords = np.floor(coords / self.cfg.voxel_size)
        coords, unique_map, inverse_map = np.unique(coords, return_index=True, return_inverse=True, axis=0)
        return coords, feature[unique_map], unique_map, inverse_map

    # ──────────────────────────────────────────────────────────────────────
    # Superpoint helpers
    # ──────────────────────────────────────────────────────────────────────

    def remake_sp_idx(self, sp_idx, unique_map):
        sp_idx_voxel = sp_idx[unique_map]
        valid_mask = sp_idx_voxel != -1
        unique_vals = np.unique(sp_idx_voxel[valid_mask])
        unique_vals.sort()
        remapped = -np.ones_like(sp_idx_voxel)
        remapped[valid_mask] = np.searchsorted(unique_vals, sp_idx_voxel[valid_mask])
        return remapped, unique_vals

    def remap_nbr(self, sp_nbr, old_sp_ids, new_sp_ids):
        id_mapping = {old_id: new_id for old_id, new_id in zip(old_sp_ids, new_sp_ids)}
        new_neighbor_dict = {}
        for old_sp_id, old_neighbors in sp_nbr.items():
            if old_sp_id not in id_mapping:
                continue
            new_neighbor_dict[id_mapping[old_sp_id]] = [id_mapping[n] for n in old_neighbors if n in id_mapping]
        return new_neighbor_dict

    # ──────────────────────────────────────────────────────────────────────
    # Pseudo mask loading
    # ──────────────────────────────────────────────────────────────────────

    def empty_pseudo_mask(self, num_points):
        return [torch.zeros((num_points, 1)).bool(), torch.tensor(0).unsqueeze(-1), torch.tensor(False).unsqueeze(-1)]

    def load_exist_mask(self, scene_name, unique_map, num_points):
        path = os.path.join(self.cfg.save_path, 'exist_pseudo', scene_name + '.pth')
        if not os.path.exists(path):
            return self.empty_pseudo_mask(num_points)
        try:
            data = torch.load(path, weights_only=False, map_location='cpu')
            return [data['mask'][unique_map], data['score'], data['domain']]
        except Exception:
            print('removing:', path)
            os.remove(path)
            return self.empty_pseudo_mask(num_points)

    def load_prexist_mask(self, scene_name, unique_map):
        path = os.path.join(self.cfg.pre_pseudo, scene_name + '.pth')
        if not os.path.exists(path):
            return None
        predata = torch.load(path, weights_only=False, map_location='cpu')
        return [predata['mask'][unique_map], torch.ones_like(predata['score']) * 100]

    # ──────────────────────────────────────────────────────────────────────
    # DataLoader
    # ──────────────────────────────────────────────────────────────────────

    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn, shuffle=shuffle, drop_last=True, pin_memory=True,
            persistent_workers=(self.mode == 'train' and self.cfg.num_workers > 0))

    def collate_fn(self, batch):
        samples = {key: tuple(sample[key] for sample in batch) for key in batch[0]}
        has_seg = samples["coords_seg"][0] is not None
        batch_size = len(samples["coords"])

        coords_batch, feature_batch = [], []
        semantic_batch, instance_batch = [], []
        voxel_pc_batch, raw_feature_batch = [], []
        voxel_sp_batch, exist_mask_batch = [], []
        coords_seg_batch, feature_seg_batch = [], []
        target = []
        batch_num_points = 0

        for b in range(batch_size):
            num_points = samples["coords"][b].shape[0]
            batch_num_points += num_points
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                print(f'Truncating batch at {b}/{batch_size} ({batch_num_points - num_points} pts)')
                break

            coords_batch.append(torch.cat((torch.full((num_points, 1), b, dtype=torch.int32),
                                           torch.from_numpy(samples["coords"][b]).int()), 1))
            feature_batch.append(samples["feature"][b])
            voxel_pc_batch.append(samples["voxel_pc"][b])
            raw_feature_batch.append(samples["raw_feature"][b])
            voxel_sp_batch.append(samples["voxel_sp"][b])
            exist_mask_batch.append(samples["exist_mask"][b])

            if has_seg:
                seg_pts = samples["coords_seg"][b].shape[0]
                coords_seg_batch.append(torch.cat((torch.full((seg_pts, 1), b, dtype=torch.int32),
                                                   torch.from_numpy(samples["coords_seg"][b]).int()), 1))
                feature_seg_batch.append(samples["feature_seg"][b])

            semantic_i = samples["semantic"][b]
            instance_i = samples["instance"][b]
            instance_batch.append(instance_i + semantic_i * 1000)
            semantic_batch.append(semantic_i)

            instance_ids = torch.unique(instance_i)
            instance_ids = instance_ids[instance_ids != -1]
            if instance_ids.numel() == 0:
                target.append({"labels": [], "masks": torch.zeros_like(instance_i)[None, :]})
            else:
                masks = instance_ids[:, None] == instance_i[None, :]
                labels = torch.stack([torch.mode(semantic_i[m]).values for m in masks]).long()
                keep = labels >= 0
                if keep.any():
                    target.append({"labels": labels[keep], "masks": masks[keep]})
                else:
                    target.append({"labels": [], "masks": torch.zeros_like(instance_i)[None, :]})

        return {"coords": coords_batch, "feature": feature_batch, "target": target,
            "scene_name": samples["scene_name"], "semantic": semantic_batch, "instance": instance_batch,
            "inverse_map": samples["inverse_map"], "unique_map": samples["unique_map"],
            "voxel_pc": voxel_pc_batch, "raw_feature": raw_feature_batch,
            "voxel_sp": voxel_sp_batch, "sp_nbrs": samples["sp_nbrs"],
            "exist_mask": exist_mask_batch, "prexist_mask": samples["prexist_mask"],
            "dino": samples["dino"], "dist_adj": samples["dist_adj"],
            "coords_seg": torch.cat(coords_seg_batch, 0).int() if has_seg else None,
            "feature_seg": torch.cat(feature_seg_batch, 0).float() if has_seg else None,
            "inverse_map_seg": samples["inverse_map_seg"],
            "sp_idx_voxel_seg": samples["sp_idx_voxel_seg"]}
