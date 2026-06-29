import colorsys
import functools
import os
import random
import time
from collections import namedtuple
from dataclasses import dataclass, field
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Bernoulli
from torch.distributions.categorical import Categorical
from torch_scatter import scatter_mean, scatter_sum

import spconv.pytorch as spconv
from benchmark.evaluate_semantic_instance import evaluate as muti_eval
from lib.helper_ply import write_ply
from mask3d_spconv.matcher_tmp import HungarianMatcher
from trellis_tools.eval_newcf import compute_cf, convert_point2ss, normalize_point_cloud

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'logprob', 'td_target', 'value', 'advantage'))


@dataclass
class TrajectoryState:
    bs: int
    env_sp_idx: torch.Tensor
    done: List[bool] = field(default_factory=lambda: [False])
    cur_sp_idx: list = field(default_factory=list)
    cur_sp2point_mask: list = field(default_factory=list)
    nbr_sp_idx: list = field(default_factory=list)
    sample_idx: list = field(default_factory=list)
    traj: list = field(default_factory=list)
    consist: list = field(default_factory=list)
    cf: list = field(default_factory=list)
    score_2d: list = field(default_factory=list)
    score_3d: list = field(default_factory=list)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory, self.adv = [], []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.adv.append(None)
        self.memory[self.position] = Transition(*args)
        self.adv[self.position] = args[6].squeeze().item()
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def adv_mean_std(self):
        adv = np.array(self.adv)
        return adv.mean(), adv.std()

    def __len__(self):
        return len(self.memory)


class Trainer:
    def __init__(self, model, pponet, cf_model, logger, train_dataset, val_dataset, save_path, cfg=None):
        self.model = model.cuda()
        self.cf_encoder = cf_model[0].eval().cuda()
        self.cf_decoder = cf_model[1].eval().cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.logger = logger
        self.cfg = cfg
        self.matcher = HungarianMatcher()
        self.pponet = pponet.cuda()
        self.ppo_optimizer = torch.optim.AdamW(self.pponet.parameters(), lr=cfg.lr)
        self.scaler = GradScaler()

    def train_model(self, epochs):
        train_data_loader = self.train_dataset.get_loader(shuffle=True)
        start = self.load_checkpoint()
        self.refresh_info()
        for epoch in range(start, epochs):
            for batch_idx, batch in enumerate(train_data_loader):
                self.train_batch(batch, batch_idx + 1, epoch, len(train_data_loader))
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
                self.validation(vis=False, log=True)

    def train_batch(self, batch, batch_idx, epoch, loader_size):
        t0 = time.time()
        self._prepare_batch(batch)
        self._infer_features()
        self.infer_time += time.time() - t0

        t1 = time.time()
        step_R, step_num, pseudo = self._rollout()
        self._compute_gae_and_store()
        pseudo_mask_list, valid_bs = self._update_pseudo_masks(pseudo)
        self.collect_time += time.time() - t1

        t2 = time.time()
        self._optimize(pseudo_mask_list, valid_bs)
        self.optimize_time += time.time() - t2

        self.training_iter += 1
        self.step_reward += step_R / step_num
        self.traj_length += step_num / len(self._all_initsp_idx)
        if self.training_iter % self.logging_interval == 0:
            self._log_epoch(epoch, batch_idx, loader_size)

    def _prepare_batch(self, batch):
        self._target = batch["target"]
        self._scene_name = batch["scene_name"]
        self._instance = batch["instance"]
        self._inverse_map = batch["inverse_map"]
        self._unique_map = batch["unique_map"]
        self._sp_nbrs = batch["sp_nbrs"]
        self._exist_pseudo = batch["exist_mask"]
        self._prexist_pseudo = batch["prexist_mask"]
        self._inv_seg = batch["inverse_map_seg"]

        feature = batch["feature"]
        self._coords_list = [feature[i][:, 3:].cuda() for i in range(len(feature))]
        coords = torch.cat(batch["coords"], 0).int().contiguous()
        self._feature = torch.cat(feature, 0).float()
        self._batch_sp = [batch["voxel_sp"][i].cuda() for i in range(len(batch["voxel_sp"]))]
        self._pc = [batch["voxel_pc"][i].cuda() for i in range(len(batch["voxel_sp"]))]
        self._bs = len(self._pc)

        self._in_field = spconv.SparseConvTensor(
            features=self._feature.cuda(), indices=coords.cuda(),
            spatial_shape=list(coords.max(0)[0] + 16)[1:],
            batch_size=coords.max(0)[0][0].int().item() + 1)

        sp_size = [scatter_sum(torch.ones_like(self._batch_sp[b]), self._batch_sp[b], dim=0) for b in range(self._bs)]
        self._num_sp = [sp_size[b].numel() for b in range(self._bs)]
        sp_height = [scatter_mean(self._pc[b][:, -1], self._batch_sp[b], dim=0) for b in range(self._bs)]

        self._nonobj_sp_mask = [(((sp_height[b] < self.cfg.objsp_height_min) | (sp_height[b] > self.cfg.objsp_height_max))
             & (sp_size[b] > self.cfg.initsp_max_voxel)) | (sp_size[b] < self.cfg.initsp_min_voxel) for b in range(self._bs)]
        self._possible_obj_sp_idx = [torch.where(~self._nonobj_sp_mask[b])[0] for b in range(self._bs)]
        self._possible_sp_sets = [set(self._possible_obj_sp_idx[b].cpu().tolist()) for b in range(self._bs)]
        self._possible_sp_pos = []
        for b in range(self._bs):
            pos = torch.full((self._num_sp[b],), -1, dtype=torch.long, device=self._batch_sp[b].device)
            pos[self._possible_obj_sp_idx[b]] = torch.arange(self._possible_obj_sp_idx[b].numel(), device=pos.device)
            self._possible_sp_pos.append(pos)

        self._exist_sp2point_mask = [self.make_sp_point_mask(self._batch_sp[b], self._possible_obj_sp_idx[b], self._num_sp[b]) for b in range(self._bs)]
        dino = [batch["dino"][b].cuda(non_blocking=True) for b in range(self._bs)]
        batch_sp_dino = [F.normalize(dino[b], p=2, dim=1) for b in range(self._bs)]
        dist_adj = [torch.as_tensor(batch["dist_adj"][b], dtype=torch.bool, device=self._batch_sp[b].device) for b in range(self._bs)]

        self._W, self._D, self._top_min = [], [], []
        for b in range(self._bs):
            cos_b = (batch_sp_dino[b] @ batch_sp_dino[b].t()).clamp(-1.0, 1.0)
            W_b = ((cos_b > 0.8) & dist_adj[b]).float()
            W_b = W_b[self._possible_obj_sp_idx[b]][:, self._possible_obj_sp_idx[b]]
            W_b.fill_diagonal_(0.0)
            D_b = W_b.sum(dim=1)
            top_min_b = 0.5
            domain = self._exist_pseudo[b][2]
            if domain.sum() > 0 and self._exist_pseudo[b][0].shape[1] == self.cfg.topk:
                premask = self._exist_pseudo[b][0][:, torch.where(domain)[0]].cuda(non_blocking=True)
                prespmask = scatter_mean(premask.float(), self._batch_sp[b], dim=0) > 0.5
                prespmask = prespmask[self._possible_obj_sp_idx[b]]
                valid_cols = prespmask.sum(dim=0) > 2
                if valid_cols.sum() >= self.cfg.topk:
                    costs = self.compute_ncut_cost_batch(W_b, D_b, prespmask[:, valid_cols])
                    top_min_b = costs.topk(self.cfg.topk, largest=False).values[-1].item()
            self._W.append(W_b)
            self._D.append(D_b)
            self._top_min.append(top_min_b)

        self._batch_sp_seg = [batch["sp_idx_voxel_seg"][b].cuda() for b in range(len(batch["sp_idx_voxel_seg"]))]
        self._feature_seg = batch["feature_seg"].cuda(non_blocking=True)
        self._coords_seg = batch["coords_seg"]

    def _infer_features(self):
        self.model.eval()
        self.pponet.eval()
        with torch.no_grad():
            kw = dict(raw_coordinates=self._feature[:, -3:].cuda(), train_on_segments=self.cfg.use_sp,
                      env_num=self.cfg.block_num, is_datacollect=True)
            if self.cfg.use_sp:
                output = self.model(self._in_field, point2segment=self._batch_sp, **kw)
            else:
                output = self.model(self._in_field, **kw)
            self._bkb_feature = [f.detach() for f in output["mask_features"]]
            self._sp_feats = [scatter_mean(self._bkb_feature[b], self._batch_sp[b], dim=0) for b in range(self._bs)]
            self._sp_locs = [scatter_mean(self._pc[b], self._batch_sp[b], dim=0) for b in range(self._bs)]
        del output

    def _rollout(self):
        self.memory = ReplayMemory(self._bs * self.cfg.block_num * (self.cfg.max_step - 1))
        self.memory0 = ReplayMemory(self._bs * self.cfg.block_num)
        pseudo = [{"mask": [], "domain": [], "score": []} for _ in range(self._bs)]
        step_num, step_R = 0, 0

        all_initsp_loc, all_validsp_idx = [], []
        all_initsp_idx = []
        for b in range(self._bs):
            mask = (~self._nonobj_sp_mask[b]) & (scatter_mean(self._instance[b].cuda(), self._batch_sp[b], dim=0) > 0)
            candidate_sp = torch.where(mask)[0]
            candidate_sp_loc = self._sp_locs[b][candidate_sp]
            all_initsp_loc.append(candidate_sp_loc)
            all_validsp_idx.append(candidate_sp)
            all_initsp_idx.extend((b, candidate_sp_loc[i], candidate_sp[i]) for i in range(len(candidate_sp)))

        if len(all_initsp_idx) > self.cfg.trajectory_capacity:
            all_initsp_idx = random.sample(all_initsp_idx, k=self.cfg.trajectory_capacity)
        self._all_initsp_idx = all_initsp_idx

        self.trajectories = {}
        for traj_id, (bs, sp_loc, _) in enumerate(all_initsp_idx):
            sp2anchor = (all_initsp_loc[bs] - sp_loc).norm(p=2, dim=-1)
            env_sp_idx = all_validsp_idx[bs][sp2anchor <= self.cfg.env_radius]
            if len(env_sp_idx) > 10:
                self.trajectories[traj_id] = TrajectoryState(bs=bs, env_sp_idx=env_sp_idx)

        for t in range(self.cfg.max_step):
            state, not_done_traj = [], []
            for traj_id in list(self.trajectories.keys()):
                traj = self.trajectories[traj_id]
                if len(traj.done) <= t or traj.done[t]:
                    continue
                not_done_traj.append(traj_id)
                if t == 0:
                    env_sp_feats = self._sp_feats[traj.bs][traj.env_sp_idx]
                    n = len(traj.env_sp_idx)
                    si = np.random.choice(n, self.cfg.init_sp_num, replace=(n < self.cfg.init_sp_num))
                    state.append([traj_id, env_sp_feats[si], si, traj.env_sp_idx])
                else:
                    b = traj.bs
                    pt_mask = torch.where(self.make_sp_point_mask(self._batch_sp[b], traj.cur_sp_idx[-1], self._num_sp[b]))[0]
                    sp_loc = self._pc[b][pt_mask].mean(0, keepdim=True)
                    sp_feat = self._bkb_feature[b][pt_mask].mean(0, keepdim=True)
                    nbr_idx = traj.nbr_sp_idx[-1]
                    nbr_loc = self._sp_locs[b][nbr_idx]
                    nbr_feat = self._sp_feats[b][nbr_idx]
                    n = len(nbr_loc)
                    si = np.random.choice(n, self.cfg.nbr_sp_num, replace=(n < self.cfg.nbr_sp_num))
                    traj.sample_idx.append(si)
                    state.append([traj_id, sp_loc, sp_feat, nbr_loc[si], nbr_feat[si], si, pt_mask, nbr_idx])

            if not not_done_traj:
                break
            if t == 0:
                actions, logprobs, values = self.select_action0(torch.cat([s[1][None] for s in state]))
            else:
                actions, logprobs, values = self.select_action(
                    torch.cat([torch.cat((s[1], s[3]), dim=0)[None] for s in state]),
                    torch.cat([torch.cat((s[2], s[4]), dim=0)[None] for s in state]))

            for idx, traj_id in enumerate(not_done_traj):
                traj = self.trajectories[traj_id]
                action = actions[idx].long()
                if t == 0:
                    cur_sp_idx = [traj.env_sp_idx[state[idx][2][int(action.item())]].item()]
                else:
                    cur_sp_idx = self.merge_sp(action, traj.cur_sp_idx[-1], traj.nbr_sp_idx[-1], traj.sample_idx[-1])
                traj.cur_sp_idx.append(cur_sp_idx)
                traj.cur_sp2point_mask.append(self.make_sp_point_mask(self._batch_sp[traj.bs], cur_sp_idx, self._num_sp[traj.bs]))

            if t > 0:
                with torch.no_grad():
                    t_r = time.time()
                    self._compute_reward(not_done_traj)
                    self.reward_time += time.time() - t_r

            for idx, traj_id in enumerate(not_done_traj):
                traj = self.trajectories[traj_id]
                bs = traj.bs
                consist = False if t == 0 else traj.consist[-1]
                cf = False if t == 0 else traj.cf[-1]
                success = consist | cf
                sp2point_mask = traj.cur_sp2point_mask[-1]
                if success:
                    score = (traj.score_2d[-1] * traj.score_3d[-1] * 10 if consist and cf
                             else traj.score_2d[-1] if consist else traj.score_3d[-1])
                    revised_sp = (scatter_mean(sp2point_mask.float(), self._batch_sp[bs], dim=0) >= 0.5).float()
                    sp2point_mask = revised_sp[self._batch_sp[bs]]
                iou = self.get_maxmatch_mask(self._target[bs]['masks'].cuda() * self._exist_sp2point_mask[bs], sp2point_mask)[0].item()
                if success:
                    self.cf_pre += iou >= self.cfg.iou_thr
                    self.cf_count += 1
                if iou >= self.cfg.iou_thr:
                    self.cf_rec += success
                    self.cf_count2 += 1
                reward = self.cfg.success_reward if success else -1
                done = bool(success) or t == self.cfg.max_step - 1
                if not done:
                    nbr_sp_idx = self.query_dict(self._sp_nbrs[bs], traj.cur_sp_idx[-1])
                    nbrsp_filter = [s for s in nbr_sp_idx if s in self._possible_sp_sets[bs]]
                    if not nbrsp_filter:
                        done = True
                    else:
                        traj.nbr_sp_idx.append(nbrsp_filter)
                        self.nbr_count += len(nbrsp_filter)
                        self.count += 1
                traj.traj.append((state[idx], actions[idx][None], reward, logprobs[idx][None], values[idx], done))
                traj.done.append(done)
                if reward == self.cfg.success_reward:
                    pseudo[bs]["mask"].append(sp2point_mask.unsqueeze(-1))
                    pseudo[bs]["score"].append(torch.as_tensor(score).detach().float().cpu().view(1))
                    pseudo[bs]["domain"].append(bool(consist))
                    self.ious += iou
                    self.num_ious += 1
                    if iou >= 0.5:
                        self.ious50 += iou
                        self.num_ious50 += 1
                step_R += reward
                step_num += 1

        for traj_id, traj in list(self.trajectories.items()):
            if traj.traj[-1][2] != self.cfg.success_reward and random.random() <= 0.9 and len(self.trajectories) > 10:
                del self.trajectories[traj_id]
        return step_R, step_num, pseudo

    def _compute_gae_and_store(self):
        for traj in self.trajectories.values():
            trajectory = traj.traj
            if self.cfg.gae:
                lastgae = 0
                gae = torch.zeros(len(trajectory))
                for t in reversed(range(len(trajectory))):
                    nextnonterminal = 1.0 - trajectory[min(t, len(trajectory) - 1)][-1]
                    nextvalues = 0 if t == len(trajectory) - 1 else trajectory[t + 1][-2]
                    if t < len(trajectory) - 1:
                        nextnonterminal = 1.0 - trajectory[t][-1]
                    td_delta = trajectory[t][2] + self.cfg.rl_gamma * nextvalues * nextnonterminal - trajectory[t][-2]
                    gae[t] = lastgae = td_delta + self.cfg.rl_gamma * self.cfg.gae_lambda * nextnonterminal * lastgae

            for step, (state, action, reward, logprob, value, done) in enumerate(trajectory):
                next_value = trajectory[step + 1][-2] if step < len(trajectory) - 1 else 0
                if self.cfg.gae:
                    advantage, td_target = gae[step], value + gae[step]
                else:
                    td_target = reward + self.cfg.rl_gamma * next_value * (1 - done)
                    advantage = td_target - value
                (self.memory0 if step == 0 else self.memory).push(state, action, reward, logprob, td_target, value, advantage)

    def _update_pseudo_masks(self, pseudo):
        pseudo_mask_list, valid_bs = [], []
        for b in range(self._bs):
            pseudo_b = pseudo[b]
            if len(pseudo_b["mask"]) > 0:
                cur_pseudo = torch.cat([torch.cat(pseudo_b["mask"], dim=-1), self._exist_pseudo[b][0].cuda()], dim=-1)
                cur_score = torch.cat([torch.cat(pseudo_b["score"]), self._exist_pseudo[b][1]], dim=-1).cuda()
                cur_domain = torch.cat([torch.tensor(pseudo_b["domain"]), self._exist_pseudo[b][2]], dim=-1).cuda()
            else:
                cur_pseudo = self._exist_pseudo[b][0].cuda()
                cur_score = self._exist_pseudo[b][1].cuda()
                cur_domain = self._exist_pseudo[b][2].cuda()

            if (cur_pseudo.sum(0) > 2).sum() > 0:
                valid_idx = torch.where(cur_pseudo.sum(0) > 2)[0]
                cur_pseudo, cur_score, cur_domain = cur_pseudo[:, valid_idx], cur_score[valid_idx], cur_domain[valid_idx]
                # Split 2D/3D, keep top-k 2D
                t_idx, f_idx = torch.where(cur_domain == 1)[0], torch.where(cur_domain == 0)[0]
                p2d, s2d, d2d = cur_pseudo[:, t_idx], cur_score[t_idx], cur_domain[t_idx]
                p3d, s3d, d3d = cur_pseudo[:, f_idx], cur_score[f_idx], cur_domain[f_idx]
                order = torch.argsort(s2d, descending=True)
                p2d, s2d, d2d = p2d[:, order], s2d[order], d2d[order]
                if p2d.shape[1] > self.cfg.topk:
                    p2d, s2d, d2d = p2d[:, :self.cfg.topk], s2d[:self.cfg.topk], d2d[:self.cfg.topk]
                cur_pseudo = torch.cat([p2d, p3d], dim=-1)
                cur_score = torch.cat([s2d, s3d])
                cur_domain = torch.cat([d2d, d3d])
                nodup = remove_duplications(cur_pseudo, score=cur_score, iou_th=0.5)
                cur_pseudo, cur_score, cur_domain = cur_pseudo[:, nodup], cur_score[nodup], cur_domain[nodup]
                discover_num = cur_pseudo.shape[1]
                store = False
                if self._prexist_pseudo[b] is not None:
                    total_pseudo = torch.cat([cur_pseudo, self._prexist_pseudo[b][0].cuda()], dim=-1)
                    total_score = torch.cat([cur_score, self._prexist_pseudo[b][1].cuda()], dim=-1)
                    nodup = remove_duplications(total_pseudo, score=total_score, iou_th=0.5)
                    total_pseudo, total_score = total_pseudo[:, nodup], total_score[nodup]
                    if (nodup < discover_num).sum() > 0:
                        kept = nodup[nodup < discover_num]
                        cur_pseudo, cur_score, cur_domain = cur_pseudo[:, kept], cur_score[kept], cur_domain[kept]
                        store = True
                else:
                    total_pseudo, total_score = cur_pseudo, cur_score
                    store = True
                if store:
                    os.makedirs(os.path.join(self.cfg.save_path, 'exist_pseudo'), exist_ok=True)
                    torch.save({'mask': cur_pseudo.cpu()[self._inverse_map[b]].bool(), 'score': cur_score.cpu(),
                                'domain': cur_domain.cpu().bool()}, os.path.join(self.cfg.save_path, 'exist_pseudo', self._scene_name[b] + '.pth'))
                valid_bs.append(b)
            elif self._prexist_pseudo[b] is not None:
                total_pseudo = self._prexist_pseudo[b][0].cuda()
                valid_bs.append(b)
            else:
                continue
            if self.cfg.use_sp:
                pseudo_mask_list.append((scatter_mean(total_pseudo[self._inverse_map[b]].float(),
                    self._batch_sp_seg[b][self._inv_seg[b]], dim=0) >= 0.5).float())
            else:
                pseudo_mask_list.append(total_pseudo[self._inverse_map[b]][self._inv_seg[b]])
        return pseudo_mask_list, valid_bs

    def _optimize(self, pseudo_mask_list, valid_bs):
        self.model.train()
        self.pponet.train()
        in_field_seg = spconv.SparseConvTensor(
            features=self._feature_seg, indices=self._coords_seg.cuda(),
            spatial_shape=list(self._coords_seg.max(0)[0] + 16)[1:],
            batch_size=self._coords_seg.max(0)[0][0].int().item() + 1)

        for _ in range(self.cfg.batch_iter):
            self.optimizer.zero_grad()
            self.ppo_optimizer.zero_grad()
            with autocast():
                kw = dict(raw_coordinates=self._feature_seg[:, -3:], train_on_segments=self.cfg.use_sp)
                if self.cfg.use_sp:
                    out = self.model(in_field_seg, point2segment=self._batch_sp_seg, **kw)
                else:
                    out = self.model(in_field_seg, **kw)
                bkb = [out["mask_features"][b][self._inv_seg[b]][self._unique_map[b]] for b in range(self._bs)]
                ppo_loss, pg_loss, critic_loss, ent_loss = self.compute_rl_loss(bkb, self._pc, self._sp_locs, self._batch_sp)
                if valid_bs:
                    mask_loss, dice_loss, class_loss = self.compute_seg_loss(out, out["pred_masks"], pseudo_mask_list, valid_bs)
                else:
                    mask_loss = dice_loss = class_loss = ppo_loss.new_zeros(())
                seg_loss = 2 * class_loss + 5 * mask_loss + 2 * dice_loss
                loss = ppo_loss + seg_loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.step(self.ppo_optimizer)
                self.scaler.update()

            self.loss_dict['loss'] += loss.item()
            self.loss_dict['ppo loss'] += ppo_loss.item()
            self.loss_dict['actor loss'] += pg_loss.item()
            self.loss_dict['critic loss'] += critic_loss.item()
            self.loss_dict['ent loss'] += ent_loss.item()
            self.loss_dict['seg loss'] += seg_loss.item()
            self.loss_dict['mask loss'] += mask_loss.item()
            self.loss_dict['dice loss'] += dice_loss.item()
            self.loss_dict['class loss'] += class_loss.item()

    def compute_seg_loss(self, output, pred_masks, pseudo_mask_list, valid_bs):
        all_outputs = [(pred_masks, output["pred_logits"], self.matcher(output, pseudo_mask_list, valid_bs))]
        for aux in output.get("aux_outputs", []):
            all_outputs.append((aux["pred_masks"], aux["pred_logits"], self.matcher(aux, pseudo_mask_list, valid_bs)))
        loss_mask = loss_dice = loss_class = 0
        for pm, pl, matchings in all_outputs:
            m, d, c = self._matched_seg_losses(pm, pl, matchings, pseudo_mask_list, valid_bs)
            loss_mask += m; loss_dice += d; loss_class += c
        return loss_mask, loss_dice, loss_class

    def _matched_seg_losses(self, pred_masks, pred_logits, matchings, pseudo_mask_list, valid_bs):
        loss_mask = loss_dice = 0
        for i, bs in enumerate(valid_bs):
            pred_idx, target_idx = matchings[i]
            n = len(pred_idx)
            pred = pred_masks[bs][:, pred_idx].t()
            target = pseudo_mask_list[i][:, target_idx.long()].t()
            loss_mask += ce_loss_jit(pred, target, n)
            loss_dice += dice_loss_jit(pred, target, n)
        target_classes = torch.full(pred_logits[valid_bs].shape[:-1], self.model.num_classes - 1,
            dtype=torch.int64, device=pred_logits.device)
        for i, _ in enumerate(valid_bs):
            target_classes[i, matchings[i][0].long()] = 0
        loss_class = F.cross_entropy(pred_logits[valid_bs].transpose(1, 2), target_classes, ignore_index=-1)
        return loss_mask, loss_dice, loss_class

    def compute_rl_loss(self, bkb_feature, pc, sp_locs, batch_sp):
        device = torch.device("cuda")
        sp_feats = [scatter_mean(bkb_feature[b], batch_sp[b], dim=0) for b in range(len(batch_sp))]

        batch, act, lp, val, td, adv = self._sample_transitions(self.memory, device)
        traj_ids = [batch.state[i][0] for i in range(len(batch.state))]
        locs, feats = [], []
        for i, tid in enumerate(traj_ids):
            bs = self.trajectories[tid].bs
            sp_mask, nbr_idx, si = batch.state[i][6], batch.state[i][7], batch.state[i][5]
            cur_loc = pc[bs][sp_mask].mean(0, keepdim=True)
            cur_feat = bkb_feature[bs][sp_mask].mean(0, keepdim=True)
            locs.append(torch.cat((cur_loc, sp_locs[bs][nbr_idx][si]))[None])
            feats.append(torch.cat((cur_feat, sp_feats[bs][nbr_idx][si]))[None])

        batch0, act0, lp0, val0, td0, adv0 = self._sample_transitions(self.memory0, device)
        feats0 = []
        for i, tid in enumerate([batch0.state[j][0] for j in range(len(batch0.state))]):
            bs = self.trajectories[tid].bs
            sp_idx, si = batch0.state[i][3], batch0.state[i][2]
            feats0.append(sp_feats[bs][sp_idx][si][None])

        logits, v = self.pponet(torch.cat(locs), torch.cat(feats), history=None)
        curr_lp = -F.binary_cross_entropy_with_logits(logits, act, reduction="none")
        logratio = (curr_lp - lp)[:, 1:].sum(-1)

        logits0, v0 = self.pponet.step0(torch.cat(feats0), history=None)
        log_prob0 = F.log_softmax(logits0, -1)
        logratio0 = log_prob0.gather(1, act0[:, None]).squeeze(1) - lp0

        logratio = torch.cat((logratio, logratio0))
        adv_all = torch.cat((adv, adv0))
        ratio = logratio.exp()
        pg_loss = -torch.min(adv_all * ratio, adv_all * ratio.clamp(1 - self.cfg.clip_actor_eps, 1 + self.cfg.clip_actor_eps)).mean()

        v_all = torch.cat((v, v0))
        td_all = torch.cat((td, td0))
        val_all = torch.cat((val, val0))
        if self.cfg.clip_value:
            v_clipped = val_all + (v_all - val_all).clamp(-self.cfg.clip_value_eps, self.cfg.clip_value_eps)
            critic_loss = 0.5 * torch.max((v_all - td_all)**2, (v_clipped - td_all)**2).mean()
        else:
            critic_loss = 0.5 * (v_all - td_all.detach()).pow(2).mean()
        bernoulli_entropy = Bernoulli(logits=logits[:, 1:]).entropy().mean()
        categorical_entropy = Categorical(logits=logits0).entropy().mean()
        ent_loss = bernoulli_entropy + categorical_entropy
        return pg_loss + critic_loss - self.cfg.ent_coeff * ent_loss, pg_loss, critic_loss, ent_loss

    def _sample_transitions(self, memory, device):
        n = min(len(memory), self.cfg.rewarder_batch_size)
        batch = Transition(*zip(*memory.sample(n)))
        act = torch.cat(batch.action).to(device)
        lp = torch.cat(batch.logprob).to(device)
        val = torch.tensor(batch.value, device=device)
        td = torch.tensor(batch.td_target, device=device)
        adv = torch.tensor(batch.advantage, device=device)
        if self.cfg.normalize_adv:
            mu, sigma = memory.adv_mean_std()
            adv = (adv - mu) / (sigma + 1e-8)
        return batch, act, lp, val, td, adv

    def _compute_reward(self, traj_id_list):
        pre_valid, batch_ss, batch_query, batch_q_len = [], [], [], []
        ncut_candidates = []
        for i, traj_id in enumerate(traj_id_list):
            traj = self.trajectories[traj_id]
            bs = traj.bs
            inmask_pc = self._coords_list[bs][traj.cur_sp2point_mask[-1]]
            if 100 < inmask_pc.shape[0] < 8000 and len(traj.cur_sp_idx[-1]) > 2:
                bbox = inmask_pc.max(0)[0] - inmask_pc.min(0)[0]
                if 0.4 < bbox[0:2].max() < 3.5 and bbox[2] < 2.5:
                    sp_mask = self.make_possible_mask(traj.cur_sp_idx[-1], self._possible_sp_pos[bs], self._W[bs].shape[0])
                    ncut_candidates.append((i, bs, sp_mask, inmask_pc))
                    continue
            ncut_candidates.append((i, bs, None, None))

        # Batched ncut by bs
        ncut_costs = {}
        by_bs = {}
        for i, bs, sp_mask, _ in ncut_candidates:
            if sp_mask is not None:
                by_bs.setdefault(bs, []).append((i, sp_mask))
        for bs, items in by_bs.items():
            masks = torch.stack([m for _, m in items], dim=1)
            costs = self.compute_ncut_cost_batch(self._W[bs], self._D[bs], masks)
            for j, (idx, _) in enumerate(items):
                ncut_costs[idx] = costs[j]

        for i, traj_id in enumerate(traj_id_list):
            traj = self.trajectories[traj_id]
            bs = traj.bs
            _, _, sp_mask, inmask_pc = ncut_candidates[i]
            dino_reward = False
            if sp_mask is not None:
                cost = ncut_costs[i]
                if cost <= self._top_min[bs]:
                    dino_reward = True
                if inmask_pc.shape[0] < 6000:
                    pre_valid.append(True)
                    norm_pc, _, _ = normalize_point_cloud(inmask_pc)
                    batch_ss.append(convert_point2ss(norm_pc))
                    batch_q_len.append(norm_pc.shape[0])
                    batch_query.append(norm_pc.bfloat16())
                else:
                    pre_valid.append(False)
            else:
                pre_valid.append(False)
            if dino_reward:
                traj.consist.append(True)
                traj.score_2d.append(1.0 - cost.detach())
            else:
                traj.consist.append(False)

        if batch_ss:
            center_nums, _, score = compute_cf(torch.cat(batch_ss).unsqueeze(1), self.cf_encoder, self.cf_decoder,
                                               batch_query, batch_q_len, None)
        counter = 0
        for i, traj_id in enumerate(traj_id_list):
            traj = self.trajectories[traj_id]
            if pre_valid[i]:
                traj.cf.append(center_nums[counter] == 1)
                traj.score_3d.append(score[counter])
                counter += 1
            else:
                traj.cf.append(False)

    def compute_ncut_cost_batch(self, W, D, masks):
        M = masks.float()
        WM = W @ M
        cuts = ((1 - M) * WM).sum(dim=0)
        vol_R = (M * D[:, None]).sum(dim=0)
        vol_notR = ((1 - M) * D[:, None]).sum(dim=0)
        return cuts / (torch.minimum(vol_R, vol_notR) + 1e-10)

    def select_action0(self, feats):
        with torch.no_grad():
            logits, value = self.pponet.step0(feats, history=None)
            dist = Categorical(logits=F.log_softmax(logits, -1).cpu())
            action = dist.sample()
            return action, dist.log_prob(action), value.detach()

    def select_action(self, loc, feats):
        with torch.no_grad():
            logits, value = self.pponet(loc, feats, history=None)
            prob = F.sigmoid(logits).detach()
            action = torch.bernoulli(prob)
            return action, -F.binary_cross_entropy_with_logits(logits, action, reduction="none"), value.detach()

    def select_best_action0(self, feats):
        with torch.no_grad():
            logits, _ = self.pponet.step0(feats, history=None)
            return F.softmax(logits, -1).cpu().argmax(1)

    def select_best_action(self, loc, feats):
        with torch.no_grad():
            logits, _ = self.pponet(loc, feats, history=None)
            return (F.sigmoid(logits) > 0.5).long()

    def make_sp_point_mask(self, batch_sp, sp_ids, num_sp):
        sp_ids = torch.as_tensor(sp_ids, dtype=torch.long, device=batch_sp.device)
        if sp_ids.numel() == 0:
            return torch.zeros_like(batch_sp, dtype=torch.bool)
        valid = sp_ids[(sp_ids >= 0) & (sp_ids < num_sp)]
        lookup = torch.zeros(num_sp, dtype=torch.bool, device=batch_sp.device)
        lookup[valid] = True
        return lookup[batch_sp]

    def make_possible_mask(self, sp_ids, possible_sp_pos, n):
        sp_ids = torch.as_tensor(sp_ids, dtype=torch.long, device=possible_sp_pos.device)
        if sp_ids.numel() == 0:
            return torch.zeros(n, dtype=torch.bool, device=possible_sp_pos.device)
        valid = sp_ids[(sp_ids >= 0) & (sp_ids < possible_sp_pos.numel())]
        pos = possible_sp_pos[valid]
        pos = pos[pos >= 0]
        mask = torch.zeros(n, dtype=torch.bool, device=possible_sp_pos.device)
        mask[pos] = True
        return mask

    def merge_sp(self, action, cur_sp_idx, nbr_sp_idx, sample_idx):
        added = {nbr_sp_idx[i] for i, act in zip(sample_idx, action.cpu()[1:]) if act == 1}
        return list(set(cur_sp_idx) | added)

    def query_dict(self, d, key_list):
        if len(key_list) == 1:
            return d[key_list[0]]
        out = set()
        for k in key_list:
            out.update(d[k])
        return list(out - set(key_list))

    def get_maxmatch_mask(self, target_mask, cur_mask):
        inter = (cur_mask[None, :] * target_mask).sum(1)
        union = cur_mask.sum() + target_mask.sum(1) - inter
        ious = inter / (union + 1e-5)
        maxiou, idx = ious.max(0)
        return maxiou, target_mask[idx]

    def refresh_info(self):
        self.loss_dict = {k: 0 for k in ['loss', 'ppo loss', 'actor loss', 'critic loss', 'ent loss',
                                           'seg loss', 'mask loss', 'dice loss', 'class loss']}
        self.training_iter = 0
        self.logging_interval = len(self.train_dataset.get_loader(shuffle=True))
        self.step_reward = self.traj_length = 0
        self.infer_time = self.collect_time = self.optimize_time = self.reward_time = 0
        self.ious = self.ious50 = 0
        self.num_ious = self.num_ious50 = 0
        self.nbr_count = self.count = 0
        self.cf_count = self.cf_count2 = 1
        self.cf_pre = self.cf_rec = 0

    def _log_epoch(self, epoch, batch_idx, loader_size):
        for k in self.loss_dict:
            self.loss_dict[k] /= self.logging_interval * self.cfg.batch_iter
        self.step_reward /= self.logging_interval
        self.traj_length /= self.logging_interval
        self.logger.info(
            '{} Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.3f}, ppo: {:.3f}, actor: {:.3f}, critic: {:.3f}, '
            'ent: {:.3f}, StepR: {:.2f}, seg: {:.3f}, mask: {:.3f}, dice: {:.3f}, class: {:.3f}, lr: {:.3e}, '
            'Traj: {:.2f}, infer time: {:.1f}s, collect time: {:.1f}s, reward time: {:.1f}s, '
            'optimize time: {:.1f}s, Elapsed time: {:.1f}s ({} iters)'.format(
                time.strftime("%Y-%m-%d %H:%M:%S"), epoch, batch_idx, loader_size,
                100. * batch_idx / loader_size, epoch * loader_size + batch_idx,
                self.loss_dict['loss'], self.loss_dict['ppo loss'], self.loss_dict['actor loss'],
                self.loss_dict['critic loss'], self.loss_dict['ent loss'], self.step_reward,
                self.loss_dict['seg loss'], self.loss_dict['mask loss'], self.loss_dict['dice loss'],
                self.loss_dict['class loss'], self.optimizer.param_groups[0]['lr'], self.traj_length,
                self.infer_time, self.collect_time, self.reward_time, self.optimize_time,
                self.infer_time + self.collect_time + self.optimize_time, self.logging_interval))
        self.logger.info('50iou percent: {:.3f}, AVG iou: {:.3f}, AVG 50iou: {:.3f}, CF Pre/Rec: {:.3f}/{:.3f})'.format(
            self.num_ious50 / (self.num_ious + 1e-5), self.ious / (self.num_ious + 1e-5),
            self.ious50 / (self.num_ious50 + 1e-5), self.cf_pre / self.cf_count, self.cf_rec / self.cf_count2))
        self.refresh_info()

    def save_checkpoint(self, epoch):
        path = os.path.join(self.save_path, f'checkpoint_{epoch}.tar')
        if not os.path.exists(path):
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'opt_model_state_dict': self.optimizer.state_dict(),
                        'ppo_state_dict': self.pponet.state_dict(),
                        'opt_ppo_state_dict': self.ppo_optimizer.state_dict()}, path)

    def load_checkpoint(self):
        checkpoints = glob(self.save_path + '/*tar')
        if not checkpoints:
            print(f'No checkpoints found at {self.save_path}')
            return 0
        epochs = sorted(int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]) for p in checkpoints)
        path = os.path.join(self.save_path, f'checkpoint_{epochs[-1]}.tar')
        print(f'Loaded checkpoint from: {path}')
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['opt_model_state_dict'])
        self.pponet.load_state_dict(ckpt['ppo_state_dict'])
        self.ppo_optimizer.load_state_dict(ckpt['opt_ppo_state_dict'])
        return ckpt['epoch'] + 1

    def validation(self, vis=True, log=False):
        self.load_checkpoint()
        self.refresh_info()
        self.preds, self.gt, self.sem_preds = {}, {}, {}
        self.model.eval()
        for batch in self.val_dataset.get_loader(shuffle=False):
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
                    output = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:].cuda(), train_on_segments=True)
                    voxel_masks = output["pred_masks"][0][voxl_sp[0]].sigmoid()
                else:
                    output = self.model(in_field, raw_coordinates=feature[:, -3:].cuda(), train_on_segments=False)
                    voxel_masks = output["pred_masks"][0].sigmoid()
                masks = voxel_masks[inverse_map].detach().cpu()
                hard_masks = masks > 0.5
                valid_idx, scores = [], []
                for mid in range(self.model.num_queries):
                    if hard_masks[:, mid].sum() > self.cfg.mask_min_voxel:
                        valid_idx.append(mid)
                        scores.append((masks[:, mid][hard_masks[:, mid]].mean() * F.softmax(output["pred_logits"][0], dim=1)[mid, 0]).item())
                valid_masks = hard_masks[:, valid_idx]

            if vis and valid_idx:
                self._visualize(scene_name, batch["raw_feature"][0][:, 3:].numpy(), valid_masks, batch["target"][0], inverse_map, 'vis_preds')

            self.preds[scene_name] = {"pred_masks": valid_masks.numpy(), "pred_scores": np.array(scores), "pred_classes": np.ones(len(valid_idx))}
            self.gt[scene_name] = os.path.join(self.cfg.data_root, 'instance_gt', self.val_dataset.mode, scene_name + '.txt')
            pred_sem = [torch.mode(semantic[inverse_map][torch.where(valid_masks[:, i])[0]]).values.item() for i in range(valid_masks.shape[1])]
            self.sem_preds[scene_name] = {"pred_masks": valid_masks.numpy(), "pred_scores": np.array(scores),
                "pred_classes": self.val_dataset.remap_model_output(np.array(pred_sem) + self.val_dataset.label_offset)}
        muti_eval(False, self.preds, self.gt, self.logger, log, self.save_path)
        muti_eval(True, self.sem_preds, self.gt, self.logger, log, self.save_path)

    def validation_pseudo(self, vis=True, log=False):
        self.refresh_info()
        self.preds, self.gt, self.sem_preds = {}, {}, {}
        self.model.eval()
        for batch in self.train_dataset.get_loader(shuffle=False):
            with torch.no_grad():
                scene_name = batch["scene_name"][0]
                semantic = batch["semantic"][0]
                inverse_map = batch["inverse_map"][0]
                exist_pseudo, prexist_pseudo = batch["exist_mask"], batch["prexist_mask"]
                cur_pseudo = torch.as_tensor(exist_pseudo[0][0])
                cur_score = exist_pseudo[0][1]
                if prexist_pseudo[0] is not None:
                    cur_pseudo = torch.cat((cur_pseudo, prexist_pseudo[0][0]), 1)
                    cur_score = torch.cat((cur_score, prexist_pseudo[0][1]))
                valid_masks = cur_pseudo[inverse_map].detach()
                mask_score = cur_score

            if vis and valid_masks.shape[1] > 0:
                self._visualize(scene_name, batch["raw_feature"][0][:, 3:].numpy(), valid_masks, batch["target"][0], inverse_map, 'vis_pseudo')

            self.preds[scene_name] = {"pred_masks": valid_masks.numpy(), "pred_scores": torch.as_tensor(mask_score).numpy(), "pred_classes": np.ones(valid_masks.shape[-1])}
            self.gt[scene_name] = os.path.join(self.cfg.data_root, 'instance_gt', 'train', scene_name + '.txt')
            pred_sem = []
            for i in range(valid_masks.shape[1]):
                if valid_masks[:, i].sum() > 0:
                    pred_sem.append(torch.mode(semantic[inverse_map][torch.where(valid_masks[:, i])[0]]).values.item())
                else:
                    pred_sem.append(-1)
            self.sem_preds[scene_name] = {"pred_masks": valid_masks.numpy(), "pred_scores": torch.as_tensor(mask_score).numpy(),
                "pred_classes": self.val_dataset.remap_model_output(np.array(pred_sem) + self.val_dataset.label_offset)}
        muti_eval(False, self.preds, self.gt, self.logger, log, self.save_path)
        muti_eval(True, self.sem_preds, self.gt, self.logger, log, self.save_path)

    def rl_inference_demo(self, output_dir=None, max_scenes=None, max_trajectories=20):
        self.load_checkpoint()
        self.model.eval()
        self.pponet.eval()
        output_dir = output_dir or os.path.join(self.save_path, 'rl_demo')
        os.makedirs(output_dir, exist_ok=True)

        loader = self.train_dataset.get_loader(shuffle=False)
        scene_count = 0

        for batch in loader:
            if max_scenes and scene_count >= max_scenes:
                break
            with torch.no_grad():
                self._prepare_batch(batch)
                self._infer_features()

                for b in range(self._bs):
                    scene_name = self._scene_name[b]
                    scene_dir = os.path.join(output_dir, scene_name)
                    full_pc = self._coords_list[b].cpu().numpy()
                    full_color = batch["raw_feature"][b][self._unique_map[b], :3].numpy().astype(np.uint8)

                    candidate_sp = torch.where(~self._nonobj_sp_mask[b])[0]
                    candidate_sp_loc = self._sp_locs[b][candidate_sp]

                    if len(candidate_sp) > max_trajectories:
                        sel = torch.randperm(len(candidate_sp))[:max_trajectories]
                        init_sps = candidate_sp[sel]
                        init_locs = candidate_sp_loc[sel]
                    else:
                        init_sps = candidate_sp
                        init_locs = candidate_sp_loc

                    all_trajectories = []

                    for traj_i in range(len(init_sps)):
                        sp_loc = init_locs[traj_i]
                        sp2anchor = (candidate_sp_loc - sp_loc).norm(p=2, dim=-1)
                        env_sp_idx = candidate_sp[sp2anchor <= self.cfg.env_radius]
                        if len(env_sp_idx) <= 10:
                            continue

                        step_masks = []
                        cur_sp_idx_history = []

                        env_sp_feats = self._sp_feats[b][env_sp_idx]
                        n = len(env_sp_idx)
                        si = np.arange(min(n, self.cfg.init_sp_num))
                        feats_input = env_sp_feats[si][None]
                        action = self.select_best_action0(feats_input)
                        cur_sp_idx = [env_sp_idx[si[int(action.item())]].item()]
                        cur_sp_idx_history.append(list(cur_sp_idx))

                        mask = self.make_sp_point_mask(self._batch_sp[b], cur_sp_idx, self._num_sp[b])
                        step_masks.append(mask.cpu())

                        for t in range(1, self.cfg.max_eval_step):
                            nbr_sp_idx = self.query_dict(self._sp_nbrs[b], cur_sp_idx)
                            nbrsp_filter = [s for s in nbr_sp_idx if s in self._possible_sp_sets[b]]
                            if not nbrsp_filter:
                                break

                            pt_mask = torch.where(mask)[0]
                            sp_loc_t = self._pc[b][pt_mask].mean(0, keepdim=True)
                            sp_feat_t = self._bkb_feature[b][pt_mask].mean(0, keepdim=True)
                            nbr_loc = self._sp_locs[b][nbrsp_filter]
                            nbr_feat = self._sp_feats[b][nbrsp_filter]
                            nn = len(nbrsp_filter)
                            si = np.arange(min(nn, self.cfg.nbr_sp_num))

                            loc_input = torch.cat((sp_loc_t, nbr_loc[si]))[None]
                            feat_input = torch.cat((sp_feat_t, nbr_feat[si]))[None]
                            action = self.select_best_action(loc_input, feat_input)
                            cur_sp_idx = self.merge_sp(action[0], cur_sp_idx, nbrsp_filter, si)
                            cur_sp_idx_history.append(list(cur_sp_idx))

                            mask = self.make_sp_point_mask(self._batch_sp[b], cur_sp_idx, self._num_sp[b])
                            step_masks.append(mask.cpu())

                        if not step_masks:
                            continue

                        # Check reward: geometry + NCut/CF
                        final_mask = mask
                        inmask_pc = self._coords_list[b][final_mask]
                        success = False
                        if 100 < inmask_pc.shape[0] < 8000 and len(cur_sp_idx) > 2:
                            bbox = inmask_pc.max(0)[0] - inmask_pc.min(0)[0]
                            if 0.4 < bbox[0:2].max() < 3.5 and bbox[2] < 2.5:
                                sp_mask = self.make_possible_mask(cur_sp_idx, self._possible_sp_pos[b], self._W[b].shape[0])
                                cost = self.compute_ncut_cost_batch(self._W[b], self._D[b], sp_mask[:, None])[0]
                                if cost <= self._top_min[b]:
                                    success = True
                                if not success and inmask_pc.shape[0] < 6000:
                                    norm_pc, _, _ = normalize_point_cloud(inmask_pc)
                                    ss = convert_point2ss(norm_pc)
                                    center_nums, _, _ = compute_cf(
                                        ss.unsqueeze(0).unsqueeze(1), self.cf_encoder, self.cf_decoder,
                                        [norm_pc.bfloat16()], [norm_pc.shape[0]], None)
                                    if center_nums[0] == 1:
                                        success = True

                        if not success:
                            continue

                        # Check IoU with GT
                        gt_masks = self._target[b]['masks']
                        if len(gt_masks) == 0:
                            continue
                        revised_sp = (scatter_mean(final_mask.float(), self._batch_sp[b], dim=0) >= 0.5).float()
                        final_mask_revised = revised_sp[self._batch_sp[b]]
                        iou = self.get_maxmatch_mask(
                            gt_masks.cuda() * self._exist_sp2point_mask[b], final_mask_revised)[0].item()
                        if iou < self.cfg.iou_thr:
                            continue

                        all_trajectories.append(step_masks)

                    # Visualize only successful trajectories
                    if all_trajectories:
                        os.makedirs(scene_dir, exist_ok=True)
                    traj_colors = get_evenly_distributed_colors(max(len(all_trajectories), 2))
                    for traj_i, step_masks in enumerate(all_trajectories):
                        traj_dir = os.path.join(scene_dir, f'traj_{traj_i:03d}')
                        os.makedirs(traj_dir, exist_ok=True)
                        color = traj_colors[traj_i]

                        for t, smask in enumerate(step_masks):
                            pc_color = full_color.copy()
                            pc_color[smask.numpy()] = color
                            write_ply(os.path.join(traj_dir, f'step_{t}.ply'),
                                      [full_pc, pc_color], ['x', 'y', 'z', 'red', 'green', 'blue'])

                        pc_color = full_color.copy()
                        for smask in step_masks:
                            pc_color[smask.numpy()] = color
                        write_ply(os.path.join(traj_dir, 'final.ply'),
                                  [full_pc, pc_color], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    print(f'{scene_name}: {len(all_trajectories)} successful trajectories (IoU>=0.5)')
            scene_count += 1

    def _visualize(self, scene_name, full_pc, masks, target, inverse_map, subdir):
        full_pc = full_pc.copy() - full_pc.mean(0)
        out_dir = os.path.join(self.cfg.save_path, subdir)
        os.makedirs(out_dir, exist_ok=True)
        if masks.shape[1] > 0:
            colors = np.vstack(get_evenly_distributed_colors(masks.shape[1]))
            pc_color = np.full_like(full_pc, 128, dtype=np.uint8)
            for i in range(masks.shape[1]):
                pc_color[masks[:, i]] = colors[i]
            write_ply(os.path.join(out_dir, scene_name + 'preds.ply'), [full_pc, pc_color], ['x', 'y', 'z', 'red', 'green', 'blue'])
        if len(target['masks']) > 0:
            gt_colors = np.vstack(get_evenly_distributed_colors(len(target['masks'])))
            gt_color = np.full_like(full_pc, 128, dtype=np.uint8)
            for i in range(len(target['masks'])):
                gt_color[target['masks'][:, inverse_map][i] == 1] = gt_colors[i]
            write_ply(os.path.join(out_dir, scene_name + 'gt.ply'), [full_pc, gt_color], ['x', 'y', 'z', 'red', 'green', 'blue'])


@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8), HSV_tuples))


def remove_duplications(masks, score, iou_th=0.5, inclusion_flag=True, inclusion_th=0.8):
    N = masks.shape[-1]
    active = torch.ones(N, device=masks.device)
    for i in range(N):
        if active[i] == 0:
            continue
        B = masks * active[None, :]
        inter = torch.logical_and(B, B[:, i:i+1])
        union = torch.logical_or(B, B[:, i:i+1])
        iou = inter.sum(0).float() / (union.sum(0).float() + 1e-6)
        dup = iou >= iou_th
        if dup.sum() > 1:
            s = score.clone(); s[~dup] = 0.0
            active[dup] = 0.0; active[s.argmax()] = 1.0
    if inclusion_flag:
        for i in range(N):
            if active[i] == 0:
                continue
            B = masks * active[None, :]
            inter = torch.logical_and(B, B[:, i:i+1])
            ratio = inter.sum(0).float() / (B[:, i:i+1].sum().float() + 1e-6)
            ratio[i] = 0.0
            if ratio.max() > inclusion_th:
                active[i] = 0.0
    return torch.where(active == 1)[0]


def compute_dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).sum() / num_masks


def compute_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction="none").mean(1).sum() / num_masks


dice_loss_jit = torch.jit.script(compute_dice_loss)
ce_loss_jit = torch.jit.script(compute_sigmoid_ce_loss)
