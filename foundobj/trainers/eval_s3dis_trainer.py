import colorsys
import functools
import os
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import spconv.pytorch as spconv
from benchmark.evaluate_semantic_instance_s3dis_13cls import evaluate as evaluate_s3dis_13cls
from lib.helper_ply import write_ply


@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8), HSV_tuples))


class EvalS3DISTrainer:
    def __init__(self, model, logger, val_dataset, save_path, cfg):
        self.model = model.cuda()
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.logger = logger
        self.cfg = cfg

    def load_checkpoint(self, ckpt_path=None):
        if ckpt_path is None:
            checkpoints = glob(self.save_path + '/*tar')
            if not checkpoints:
                print(f'No checkpoints found at {self.save_path}')
                return 0
            epochs = sorted([int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]) for p in checkpoints])
            ckpt_path = os.path.join(self.save_path, f'checkpoint_{epochs[-1]}.tar')

        print(f'Loaded checkpoint from: {ckpt_path}')
        checkpoint = torch.load(ckpt_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return checkpoint.get('epoch', 0)

    def validation(self, vis=False, log=False, ckpt_path=None):
        self.load_checkpoint(ckpt_path)
        self.model.eval()
        preds, gt, preds_gtsemantic = {}, {}, {}

        val_loader = self.val_dataset.get_loader(shuffle=False)
        for batch in val_loader:
            with torch.no_grad():
                (coords, feature, target, scene_name, semantic, instance,
                 inverse_map, unique_map, voxl_pc, full_pc, voxl_sp,
                 pointsp, full_semantic, raw_rgbxyz) = batch

                batch_sp = [voxl_sp[i].cuda() for i in range(len(voxl_sp))]
                coords, feature = coords.int().contiguous(), feature.float()
                in_field = spconv.SparseConvTensor(
                    features=feature.cuda(), indices=coords.int().cuda(),
                    spatial_shape=list(coords.max(0)[0] + 16)[1:],
                    batch_size=coords.max(0)[0][0].int().item() + 1)

                if self.cfg.use_sp:
                    output = self.model(in_field, point2segment=batch_sp,
                                        raw_coordinates=feature[:, -3:].cuda(),
                                        train_on_segments=True)
                    voxel_masks = output["pred_masks"][0][voxl_sp[0]].sigmoid()
                else:
                    output = self.model(in_field, raw_coordinates=feature[:, -3:].cuda(),
                                        train_on_segments=False)
                    voxel_masks = output["pred_masks"][0].sigmoid()

                masks = voxel_masks[inverse_map[0]].detach().cpu()
                hard_masks = masks > 0.5

                valid_mask_idx, mask_score = [], []
                for mask_id in range(self.model.num_queries):
                    if hard_masks[:, mask_id].sum() > 0:
                        score = (masks[:, mask_id][hard_masks[:, mask_id]].mean()
                                 * F.softmax(output["pred_logits"][0], dim=1)[mask_id, 0])
                        valid_mask_idx.append(mask_id)
                        mask_score.append(score.item())

                valid_masks = hard_masks[:, valid_mask_idx]

            if vis and valid_mask_idx:
                self._visualize(scene_name[0], full_pc[0].numpy(), valid_masks, full_semantic[0])

            preds[scene_name[0]] = {
                "pred_masks": valid_masks.cpu().numpy(),
                "pred_scores": np.array(mask_score),
                "pred_classes": (8 + 1) * np.ones(len(valid_mask_idx))
            }
            gt[scene_name[0]] = os.path.join(self.cfg.data_dir, 'instance_gt', scene_name[0] + '.txt')

            gt_semantic_classes = []
            voxel_semantic = semantic[0]
            for mask_idx in range(valid_masks.shape[1]):
                mask = valid_masks[:, mask_idx]
                if mask.sum() > 0:
                    voxelized_mask = mask[unique_map[0]]
                    if voxelized_mask.sum() > 0:
                        mask_semantic = voxel_semantic[voxelized_mask]
                        gt_class = int(torch.mode(mask_semantic).values.item())
                        gt_semantic_classes.append(gt_class + 1 if gt_class != -1 else -1)
                    else:
                        gt_semantic_classes.append(-1)
                else:
                    gt_semantic_classes.append(-1)

            preds_gtsemantic[scene_name[0]] = {
                "pred_masks": valid_masks.cpu().numpy(),
                "pred_scores": np.array(mask_score),
                "pred_classes": np.array(gt_semantic_classes)
            }

        evaluate_s3dis_13cls(False, preds, gt, self.logger, log, self.save_path)
        evaluate_s3dis_13cls(True, preds_gtsemantic, gt, self.logger, log, self.save_path)

    def _visualize(self, scene_name, full_pc, valid_masks, full_semantic):
        non_ceiling_mask = (full_semantic != 0) & (full_semantic != 12)
        area_name, room_name = scene_name.split('/')[0], scene_name.split('/')[1]
        os.makedirs(os.path.join(self.save_path, 'vis', area_name), exist_ok=True)

        pred_instance_color = np.vstack(get_evenly_distributed_colors(valid_masks.shape[1]))
        predcolor = np.ones_like(full_pc) * 128
        for mask_id in range(valid_masks.shape[1]):
            predcolor[valid_masks[:, mask_id]] = pred_instance_color[mask_id]

        pc_vis = full_pc[non_ceiling_mask] - full_pc[non_ceiling_mask].mean(0)
        write_ply(
            os.path.join(self.save_path, 'vis', area_name, room_name + '_preds.ply'),
            [pc_vis, predcolor[non_ceiling_mask].astype(np.uint8)],
            ['x', 'y', 'z', 'red', 'green', 'blue'])
