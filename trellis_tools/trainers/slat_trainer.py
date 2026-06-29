import torch
import torch.nn.functional as F
# from models.structured_latent_distill_flow import SLatFlowModel
from models.structured_latent_flow import SLatFlowModel
from torch.utils.data import DataLoader
import numpy as np
import os, copy, time
from easydict import EasyDict as edict
from glob import glob

from safetensors.torch import load_file
import spconv.pytorch as spconv
from datasets.voxel2dino import Voxel2DINO

class Trainer:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device('cuda')
        # init
        self.step = 0
        self.init_models_optimizer_lr_scheduler()
        self.prepare_dataloader()
        # resume checkpoint
        self.load_ckpt()


    def init_models_optimizer_lr_scheduler(self):
        ## models
        self.denoiser = SLatFlowModel(in_channels = 6, out_channels = 384, model_channels = 384, num_blocks = 12,
            num_heads = 16, io_block_channels = [128], pe_mode = "ape", qk_rms_norm = True).to(self.device)
        ## optimizer
        self.optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=self.cfg.lr, weight_decay=0.0)
        ## EMA
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        for param in self.ema_denoiser.parameters():
                param.requires_grad = False


    def prepare_dataloader(self):
        self.trainset = Voxel2DINO(self.cfg.data_dir, min_aesthetic_score=4.5)
        self.trainloader = DataLoader(self.trainset, batch_size=self.cfg.trainer.args.batch_size_per_gpu, pin_memory=True,
            num_workers=8, drop_last=True, persistent_workers=True, shuffle=True)


    def sample_t(self, batch_size: int):
        return torch.sigmoid(torch.randn(batch_size) * self.cfg.t_schedule_std + self.cfg.t_schedule_mean)

    def diffuse(self, x_1: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x_1)

        t = t.view(-1, *[1 for _ in range(len(x_1.shape) - 1)])#torch.Size([8, 1, 1, 1, 1])
        x_t = (1 - t) * noise + t * x_1
        return x_t

    def get_v(self, x_1: torch.Tensor, noise: torch.Tensor):
        return x_1 - noise


    def normalize_pointcloud(self, pts):
        # pts: [bs, N, 3], 将点云归一化到[-0.5, 0.5]范围内
        min_coords = pts.min(axis=1, keepdim=True).values ## [bs, 1, 3]
        max_coords = pts.max(axis=1, keepdim=True).values ## [bs, 1, 3]
        # 计算中心和尺度
        center = (min_coords + max_coords) / 2
        scale = (max_coords - min_coords).max(2, keepdim=True).values  # 使用最大边长保持比例
        pts = (pts - center) / scale ## making range to (-0.5, 0.5)
        return pts


    def training_loss(self, target_feats, voxels):
        noise = torch.randn_like(target_feats)  # noise is x_0, torch.Size([8, 8, 16, 16, 16])
        t = self.sample_t(x_1.shape[0]).to(x_1.device).float()
        x_t = self.diffuse(x_1, t, noise=noise)  # torch.Size([8, 8, 16, 16, 16])

        ############################### spconv condition encoder  ###################################3
        ## cond_pc: [bs, N, 3]
        cond_pc = self.normalize_pointcloud(cond_pc) ## [curr_bs, N, 3],  (-0.5, 0.5)
        grid = ((cond_pc+0.5) * 64).long() ## ## [curr_bs, N, 3], [0, 63]
        grid = torch.clamp(grid, 0, 63)

        if not (torch.all(grid >= 0) and torch.all(grid < 64)):
            print(grid.max(), grid.min())
        assert torch.all(grid >= 0) and torch.all(grid < 64), "Some vertices are out of bounds"

        curr_batch_grid, curr_batch_feats, start, unq_grid_list, unq_fg_labels = [], [], 0, [], []
        for b in range(grid.shape[0]): ## here, I'm not sure whether unique is necessary?
            # unq_grid, unq_inv = torch.unique(grid[b], return_inverse=True, dim=0) ## unq is the grid coordinates [K, 3], inv is index
            unq_grid, unq_idx, unq_inv = np.unique(grid[b].cpu(), return_index=True, return_inverse=True, axis=0) ## unq is the grid coordinates [K, 3], inv is index
            unq_grid, unq_idx, unq_inv = torch.from_numpy(unq_grid).long().to(self.device), torch.from_numpy(unq_idx).long().to(self.device), torch.from_numpy(unq_inv).long().to(self.device)
            unq_grid_list.append(unq_grid), unq_fg_labels.append(fg_labels[b][unq_idx])
            curr_batch_grid.append(torch.cat((torch.full((unq_grid.shape[0], 1), b).long().to(self.device), unq_grid), dim=-1))
        curr_batch_grid, curr_batch_labels = torch.cat(curr_batch_grid), torch.cat(unq_fg_labels)
        sparse_shape = list(curr_batch_grid.max(0)[0] + 1)[1:]

        ##
        curr_batch_feats = torch.ones_like(curr_batch_grid)[:, 0][:, None].float()
        ##

        cond_sparsetensor = spconv.SparseConvTensor(features=curr_batch_feats, indices=curr_batch_grid.int().contiguous(),
            spatial_shape=sparse_shape, batch_size=grid.shape[0])

        cond, fullres = self.cond_encoder(cond_sparsetensor)# SparseTensor [K, C]
        cond_feat = cond.features ## [K, 96] fp16
        ## complete dense voxel
        batch_dense_grid = []
        for b in range(grid.shape[0]): ## here, I'm not sure whether unique is necessary?
            dense_grid_ = torch.zeros((16, 16, 16, cond_feat.shape[-1]), device=self.device, dtype=cond_feat.dtype,
                                      requires_grad=cond_feat.requires_grad)
            dense_grid = dense_grid_.clone()
            bs_mask = torch.where(cond.indices[:, 0] == b)[0]
            bs_grid, bs_feat = cond.indices[:, 1:][bs_mask], cond.features[bs_mask]
            dense_grid[bs_grid[:, 0], bs_grid[:, 1], bs_grid[:, 2]] = bs_feat
            batch_dense_grid.append(dense_grid[None, ...])
        cond = torch.cat(batch_dense_grid).permute(0, 4, 1, 2, 3) ## [bs, C, 16, 16, 16]
        ######################################################################################################

        pred = self.denoiser(x_t, t * 1000, cond)  # torch.Size([8, 8, 16, 16, 16]) sparse_structure_flow.py
        assert pred.shape == noise.shape == x_1.shape
        target = self.get_v(x_1, noise)
        terms = edict()
        terms["gen_loss"] = F.mse_loss(pred, target)
        return terms


    def run_step(self, batch):
        target_feats, voxels = batch ### clean voxel, aug_pc, batch data are automatically in cuda??
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = self.training_loss(target_feats, voxels)
            l = loss['gen_loss']
        l.backward()
        ## gradient clip
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), self.cfg.max_norm)

        ## step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_ema()
        return l.item(), loss['gen_loss'].item()


    def run(self):
        loss_display, gen_loss_display = 0, 0
        time_curr = time.time()
        data_iter = iter(self.trainloader)
        self.denoiser.train()
        while self.step < self.cfg.max_steps:
            try:
                # 获取批次数据，如果用完则重新创建迭代器
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.trainloader)
                batch = next(data_iter)
            loss, loss_gen = self.run_step(batch)
            loss_display += loss
            gen_loss_display += loss_gen
            self.step += 1
            ## logging
            if self.step % self.cfg.log_interval == 0:
                loss_display /= self.cfg.log_interval
                gen_loss_display /= self.cfg.log_interval
                time_used = time.time() - time_curr
                self.logger.info(
                    'Train Iteration: {}/{}, Loss: {:.5f}, gen_loss: {:.5f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                    self.step, self.cfg.max_steps, loss_display, gen_loss_display, self.optimizer.param_groups[0]['lr'], time_used,
                    self.cfg.log_interval))
                time_curr = time.time()
                loss_display, gen_loss_display = 0, 0
            ## save
            if self.step % self.cfg.save_interval == 0:
                self.save_ckpt()


    def load_ckpt(self):
        checkpoints = glob(self.cfg.save_path+'/*tar')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.cfg.save_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0].split('step_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = os.path.join(self.cfg.save_path, 'ckpt_step_{}.tar'.format(checkpoints[-1]))

        checkpoint = torch.load(path)
        self.denoiser.load_state_dict(checkpoint['denoiser'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        print('Loaded checkpoint from: {}'.format(path))
        del checkpoint


    def save_ckpt(self):
        ckpt = {'denoiser': self.denoiser.state_dict(), 'denoiser_ema': self.ema_denoiser.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'step': self.step}
        torch.save(ckpt, os.path.join(self.cfg.save_path, f'ckpt_step_'+ str(self.step) +'.tar'))


    def update_ema(self):
        # 遍历主模型和EMA模型的参数，成对更新
        for model_param, ema_param in zip(self.denoiser.parameters(), self.ema_denoiser.parameters()):
            ema_param.data.mul_(self.cfg.ema_rate).add_(model_param.data, alpha=1.0 - self.cfg.ema_rate)
            # 确保ema_param始终处于detach状态（虽然初始化时已冻结，但再次确认）
            ema_param.detach_()