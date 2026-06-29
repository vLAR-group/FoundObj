import colorsys
import functools
import os
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import spconv.pytorch as spconv
from benchmark.evaluate_semantic_instance200 import evaluate as muti_eval200
from lib.helper_ply import write_ply


@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8), HSV_tuples))


class EvalScanNetTrainer:
    def __init__(self, model, logger, val_dataset, save_path, cfg):
        self.model = model.cuda()
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.logger = logger
        self.cfg = cfg
        self.mask_min_voxel = getattr(cfg, 'mask_min_voxel', 50)

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
        preds, gt, sem_preds = {}, {}, {}

        val_loader = self.val_dataset.get_loader(shuffle=False)
        for batch in val_loader:
            with torch.no_grad():
                scene_name = batch["scene_name"][0]
                semantic = batch["semantic"][0]
                inverse_map = batch["inverse_map"][0]
                voxl_sp = batch["voxel_sp"]
                batch_sp = [voxl_sp[i].cuda() for i in range(len(voxl_sp))]

                coords = torch.cat(batch["coords"], 0).int().contiguous()
                feature = torch.cat(batch["feature"], 0).float()
                in_field = spconv.SparseConvTensor(features=feature.cuda(), indices=coords.cuda(),
                    spatial_shape=list(coords.max(0)[0] + 16)[1:], batch_size=coords.max(0)[0][0].int().item() + 1)

                if self.cfg.use_sp:
                    output = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:].cuda(),
                                        train_on_segments=True)
                    voxel_masks = output["pred_masks"][0][voxl_sp[0]].sigmoid()
                else:
                    output = self.model(in_field, raw_coordinates=feature[:, -3:].cuda(), train_on_segments=False)
                    voxel_masks = output["pred_masks"][0].sigmoid()

                masks = voxel_masks[inverse_map].detach().cpu()
                hard_masks = masks > 0.5

                valid_idx, scores = [], []
                for mid in range(self.model.num_queries):
                    if hard_masks[:, mid].sum() > self.mask_min_voxel:
                        score = (masks[:, mid][hard_masks[:, mid]].mean() * F.softmax(output["pred_logits"][0], dim=1)[mid, 0])
                        # score = (masks[:,mid]*hard_masks[:, mid]).sum()/(hard_masks[:, mid].sum()+1e-5)*F.softmax(output["pred_logits"][0], dim=1)[mid, 0].cpu()
                        valid_idx.append(mid)
                        scores.append(score.item())

                valid_masks = hard_masks[:, valid_idx]

            if vis and valid_idx:
                self._visualize(scene_name, batch["raw_feature"][0][:, 3:].numpy(), valid_masks)

            preds[scene_name] = {"pred_masks": valid_masks.numpy(), "pred_scores": np.array(scores),
                "pred_classes": np.ones(len(valid_idx))}
            gt[scene_name] = os.path.join(self.cfg.data_root, 'instance_gt', self.val_dataset.mode, scene_name + '.txt')

            pred_sem = []
            for i in range(valid_masks.shape[1]):
                sem = torch.mode(semantic[inverse_map][torch.where(valid_masks[:, i])[0]]).values
                pred_sem.append(sem.item())
            sem_preds[scene_name] = {"pred_masks": valid_masks.numpy(), "pred_scores": np.array(scores),
                "pred_classes": self.val_dataset.remap_model_output(np.array(pred_sem) + self.val_dataset.label_offset)}

        muti_eval200(False, preds, gt, self.logger, log, self.save_path)
        muti_eval200(True, sem_preds, gt, self.logger, log, self.save_path)

    def _visualize(self, scene_name, full_pc, valid_masks):
        os.makedirs(os.path.join(self.save_path, 'vis_preds'), exist_ok=True)
        full_pc = full_pc - full_pc.mean(0)
        pred_instance_color = np.vstack(get_evenly_distributed_colors(valid_masks.shape[1]))
        predcolor = np.ones_like(full_pc) * 128
        for mask_id in range(valid_masks.shape[1]):
            predcolor[valid_masks[:, mask_id]] = pred_instance_color[mask_id]
        write_ply(os.path.join(self.save_path, 'vis_preds', scene_name + '_preds.ply'),
            [full_pc, predcolor.astype(np.uint8)],['x', 'y', 'z', 'red', 'green', 'blue'])
