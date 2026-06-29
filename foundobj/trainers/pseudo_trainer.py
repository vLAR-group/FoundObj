import colorsys
import functools
import os
import time
from glob import glob
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import spconv.pytorch as spconv
from benchmark.evaluate_semantic_instance import evaluate
from lib.helper_ply import write_ply
from mask3d_spconv.matcher_tmp import HungarianMatcher


@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8), HSV_tuples))


def compute_dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, weights=None):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if weights is not None:
        loss = loss * weights
    return loss.sum() / num_masks


def compute_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, weights=None):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.mean(1)
    if weights is not None:
        loss = loss * weights
    return loss.sum() / num_masks


class PseudoTrainer:
    def __init__(self, model, logger, train_dataset, val_dataset, save_path, cfg):
        self.model = model.cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.logger = logger
        self.cfg = cfg
        self.matcher = HungarianMatcher()
        self.scaler = GradScaler()
        self.refresh_info()

    def refresh_info(self):
        self.loss_dict = {'loss': 0, 'mask loss': 0, 'dice loss': 0, 'class loss': 0}
        self.training_iter = 0
        self.cost_time = 0

    def train_batch(self, batch, batch_idx, epoch, loader_size):
        time_cur = time.time()
        (coords, feature, target, scene_name, semantic, instance,
         inverse_map, unique_map, voxl_pc, raw_feature, voxl_sp, raw_sp) = batch

        batch_sp = [voxl_sp[i].cuda() for i in range(len(voxl_sp))]
        valid_bs, gtmasks = [], []
        for b in range(len(batch_sp)):
            valid_bs.append(b)
            gtmasks.append(target[b]['segment_mask'].t().cuda().float())

        coords = coords.contiguous() if not coords.is_contiguous() else coords
        in_field = spconv.SparseConvTensor(features=feature.cuda(), indices=coords.int().cuda(),
            spatial_shape=list(coords.max(0)[0] + 16)[1:], batch_size=coords.max(0)[0][0].int().item() + 1)

        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            if self.cfg.use_sp:
                output = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:].cuda(),
                                    train_on_segments=True)
            else:
                output = self.model(in_field, raw_coordinates=feature[:, -3:].cuda(), train_on_segments=False)

            mask_loss, dice_loss, class_loss = self._compute_loss(
                output, output["pred_masks"], gtmasks, valid_bs)
            loss = 2 * class_loss + 5 * mask_loss + 2 * dice_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.loss_dict['loss'] += loss.item()
        self.loss_dict['mask loss'] += mask_loss.item()
        self.loss_dict['dice loss'] += dice_loss.item()
        self.loss_dict['class loss'] += class_loss.item()
        self.training_iter += 1
        self.cost_time += time.time() - time_cur

        log_interval = max(loader_size // 5, 1)
        if self.training_iter % log_interval == 0:
            for key in self.loss_dict:
                self.loss_dict[key] /= log_interval
            self.logger.info(
                '{} Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.3f}, mask: {:.3f}, dice: {:.3f}, '
                'class: {:.3f}, lr: {:.3e}, time: {:.1f}s'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), epoch, batch_idx, loader_size,
                    100. * batch_idx / loader_size, self.loss_dict['loss'], self.loss_dict['mask loss'],
                    self.loss_dict['dice loss'], self.loss_dict['class loss'],
                    self.optimizer.param_groups[0]['lr'], self.cost_time))
            self.refresh_info()

    def _compute_loss(self, output, score, pseudo_mask_list, valid_bs):
        matchings = self.matcher(output, pseudo_mask_list, valid_bs)
        aux_matching = []
        if "aux_outputs" in output:
            for aux_outputs in output["aux_outputs"]:
                aux_matching.append(self.matcher(aux_outputs, pseudo_mask_list, valid_bs))

        loss_dice, loss_mask, loss_class = 0, 0, 0
        for actual_bs_index, actual_bs in enumerate(valid_bs):
            matched_slot_num = len(matchings[actual_bs_index][0])
            loss_mask += compute_sigmoid_ce_loss(
                score[actual_bs][:, matchings[actual_bs_index][0]].t(),
                pseudo_mask_list[actual_bs_index][:, matchings[actual_bs_index][1].long()].t(),
                matched_slot_num)
            loss_dice += compute_dice_loss(
                score[actual_bs][:, matchings[actual_bs_index][0]].t(),
                pseudo_mask_list[actual_bs_index][:, matchings[actual_bs_index][1].long()].t(),
                matched_slot_num)

        target_classes = torch.full(output["pred_logits"][valid_bs].shape[:-1], self.model.num_classes - 1, dtype=torch.int64,
            device=output["pred_logits"].device)
        for actual_bs_index, actual_bs in enumerate(valid_bs):
            target_classes[actual_bs_index, matchings[actual_bs_index][0].long()] = 0
        loss_class += F.cross_entropy(output["pred_logits"][valid_bs].transpose(1, 2), target_classes, ignore_index=-1)

        if "aux_outputs" in output:
            for i, aux_outputs in enumerate(output["aux_outputs"]):
                aux_mask = aux_outputs['pred_masks']
                aux_logits = aux_outputs['pred_logits']
                tmp_matching = aux_matching[i]
                for actual_bs_index, actual_bs in enumerate(valid_bs):
                    tmp_matched = len(tmp_matching[actual_bs_index][0])
                    loss_mask += compute_sigmoid_ce_loss(
                        aux_mask[actual_bs][:, tmp_matching[actual_bs_index][0]].t(),
                        pseudo_mask_list[actual_bs_index][:, tmp_matching[actual_bs_index][1].long()].t(),
                        tmp_matched)
                    loss_dice += compute_dice_loss(
                        aux_mask[actual_bs][:, tmp_matching[actual_bs_index][0]].t(),
                        pseudo_mask_list[actual_bs_index][:, tmp_matching[actual_bs_index][1].long()].t(),
                        tmp_matched)

                target_classes = torch.full(
                    aux_logits[valid_bs].shape[:-1],
                    self.model.num_classes - 1, dtype=torch.int64,
                    device=aux_logits.device)
                for actual_bs_index, actual_bs in enumerate(valid_bs):
                    target_classes[actual_bs_index, tmp_matching[actual_bs_index][0].long()] = 0
                loss_class += F.cross_entropy(
                    aux_logits[valid_bs].transpose(1, 2), target_classes, ignore_index=-1)

        return loss_mask, loss_dice, loss_class

    def train_model(self, epochs):
        train_loader = self.train_dataset.get_loader(shuffle=True)
        start = self.load_checkpoint()
        self.refresh_info()
        for epoch in range(start, epochs):
            for batch_idx, batch in enumerate(train_loader):
                self.train_batch(batch, batch_idx + 1, epoch, len(train_loader))
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
                self.validation(vis=False, log=True)

    def save_checkpoint(self, epoch):
        path = os.path.join(self.save_path, f'checkpoint_{epoch}.tar')
        if not os.path.exists(path):
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'opt_model_state_dict': self.optimizer.state_dict(),
            }, path)

    def load_checkpoint(self):
        checkpoints = glob(self.save_path + '/*tar')
        if not checkpoints:
            self.logger.info(f'No checkpoints found at {self.save_path}')
            return 0
        epochs = sorted([int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]) for p in checkpoints])
        path = os.path.join(self.save_path, f'checkpoint_{epochs[-1]}.tar')
        self.logger.info(f'Loaded checkpoint from: {path}')
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['opt_model_state_dict'])
        return checkpoint['epoch']

    def validation(self, vis=False, log=False):
        self.model.eval()
        preds, gt, sem_preds = {}, {}, {}

        val_loader = self.val_dataset.get_loader(shuffle=False)
        for batch in val_loader:
            with torch.no_grad():
                (coords, feature, target, scene_name, semantic, instance,
                 inverse_map, unique_map, voxl_pc, raw_feature, voxl_sp, raw_sp) = batch

                batch_sp = [voxl_sp[i].cuda() for i in range(len(voxl_sp))]
                coords = coords.contiguous() if not coords.is_contiguous() else coords
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

                valid_idx, scores = [], []
                for mid in range(self.model.num_queries):
                    if hard_masks[:, mid].sum() > 50:
                        score = (masks[:, mid][hard_masks[:, mid]].mean()
                                 * F.softmax(output["pred_logits"][0], dim=1)[mid, 0])
                        valid_idx.append(mid)
                        scores.append(score.item())

                valid_masks = hard_masks[:, valid_idx]

            if vis and valid_idx:
                self._visualize(scene_name[0], raw_feature[0][:, 3:].numpy(), valid_masks)

            preds[scene_name[0]] = {
                "pred_masks": valid_masks.numpy(),
                "pred_scores": np.array(scores),
                "pred_classes": np.ones(len(valid_idx)),
            }
            gt[scene_name[0]] = os.path.join(
                self.cfg.data_dir, 'instance_gt', self.val_dataset.mode, scene_name[0] + '.txt')

            pred_sem = []
            for i in range(valid_masks.shape[1]):
                sem = torch.mode(semantic[0][inverse_map[0]][torch.where(valid_masks[:, i])[0]]).values
                pred_sem.append(sem.item())
            sem_preds[scene_name[0]] = {
                "pred_masks": valid_masks.numpy(),
                "pred_scores": np.array(scores),
                "pred_classes": self.val_dataset.remap_model_output(
                    np.array(pred_sem) + self.val_dataset.label_offset),
            }

        evaluate(False, preds, gt, self.logger, log, self.save_path)
        evaluate(True, sem_preds, gt, self.logger, log, self.save_path)

    def _visualize(self, scene_name, full_pc, valid_masks):
        os.makedirs(os.path.join(self.save_path, 'vis'), exist_ok=True)
        full_pc = full_pc - full_pc.mean(0)
        pred_color = np.vstack(get_evenly_distributed_colors(valid_masks.shape[1]))
        color = np.ones_like(full_pc) * 128
        for i in range(valid_masks.shape[1]):
            color[valid_masks[:, i]] = pred_color[i]
        write_ply(
            os.path.join(self.save_path, 'vis', scene_name + '_preds.ply'),
            [full_pc, color.astype(np.uint8)],
            ['x', 'y', 'z', 'red', 'green', 'blue'])
