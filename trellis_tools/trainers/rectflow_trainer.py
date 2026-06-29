import torch
import torch.nn.functional as F
from models.sparse_structure_vae import SparseStructureEncoder
from models.sparse_structure_flow import SparseStructureFlowModel
from datasets.point_denseconv import PointDenseConv
from torch.utils.data import DataLoader
import numpy as np
import os, copy, time
from easydict import EasyDict as edict
from glob import glob
from utils.grad_clip_utils import AdaptiveGradClipper
from Point_condition.condition_pc_prepare import voxel_point

from safetensors.torch import load_file


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
        self.supervision_encoder = SparseStructureEncoder(**self.cfg.models.encoder.args).to(self.device)
        supervision_ckpt_path = "ss_enc_conv3d_16l8_fp16.safetensors"
        self.supervision_encoder.load_state_dict(load_file(supervision_ckpt_path))
        for param in self.supervision_encoder.parameters():
            param.requires_grad = False
        self.supervision_encoder.eval()

        self.cond_encoder = SparseStructureEncoder(**self.cfg.models.cond_encoder.args).to(self.device)
        self.denoiser = SparseStructureFlowModel(**self.cfg.models.denoiser.args).to(self.device)
        ## optimizer
        self.optimizer = torch.optim.AdamW([{'params': self.cond_encoder.parameters()},
                                            {'params': self.denoiser.parameters()}], lr=self.cfg.lr, weight_decay=0.0)
        ## EMA
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.ema_cond_encoder = copy.deepcopy(self.cond_encoder)
        for param in self.ema_denoiser.parameters():
            param.requires_grad = False
        for param in self.ema_cond_encoder.parameters():
            param.requires_grad = False
        ## grad clipper
        # self.grad_clip = AdaptiveGradClipper(max_norm=self.cfg.max_norm, clip_percentile=self.cfg.clip_percentile)


    def prepare_dataloader(self):
        self.trainset = PointDenseConv(self.cfg.data_dir, min_aesthetic_score=4.5)
        self.trainloader = DataLoader(self.trainset, batch_size=self.cfg.trainer.args.batch_size_per_gpu, pin_memory=True,
            num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())), drop_last=True, persistent_workers=True)


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


    def training_loss(self, x_1: torch.Tensor, cond_pc=None, fg_labels=None):
        noise = torch.randn_like(x_1)  # noise is x_0, torch.Size([8, 8, 16, 16, 16])
        t = self.sample_t(x_1.shape[0]).to(x_1.device).float()
        x_t = self.diffuse(x_1, t, noise=noise)  # torch.Size([8, 8, 16, 16, 16])
        cond_voxel = voxel_point(cond_pc, self.cfg.voxel_size).to(self.device)
        cond = self.cond_encoder(cond_voxel, sample_posterior=False)  # torch.Size([8, 8, 16, 16, 16])

        pred = self.denoiser(x_t, t * 1000, cond)  # torch.Size([8, 8, 16, 16, 16]) sparse_structure_flow.py
        assert pred.shape == noise.shape == x_1.shape
        target = self.get_v(x_1, noise)
        terms = edict()
        terms["loss"] = F.mse_loss(pred, target)
        return terms


    def run_step(self, batch):
        x_1, cond_pc, fg_labels = batch ### clean voxel, aug_pc
        # with torch.autocast(device_type="cuda"):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                grids = x_1.float().to(self.device)
                z = self.supervision_encoder(grids, sample_posterior=False)  # [B, 1, 64, 64, 64]
            ## training
            loss = self.training_loss(z, cond_pc, fg_labels)
            l = loss['loss']
        ## backward
        l.backward()

        ## gradient clip
        torch.nn.utils.clip_grad_norm_(self.cond_encoder.parameters(), self.cfg.max_norm)
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), self.cfg.max_norm)

        ## step
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Update exponential moving average
        self.update_ema()
        return l.item()


    def run(self):
        loss_display = 0
        time_curr = time.time()
        data_iter = iter(self.trainloader)
        self.cond_encoder.train()
        self.denoiser.train()
        while self.step < self.cfg.max_steps:
            try:
                # 获取批次数据，如果用完则重新创建迭代器
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.trainloader)
                batch = next(data_iter)
            loss = self.run_step(batch)
            loss_display += loss
            self.step += 1
            ## logging
            if self.step % self.cfg.log_interval == 0:
                loss_display /= self.cfg.log_interval
                time_used = time.time() - time_curr
                self.logger.info(
                    'Train Iteration: {}/{}, Loss: {:.5f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                    self.step, self.cfg.max_steps, loss_display, self.optimizer.param_groups[0]['lr'], time_used,
                    self.cfg.log_interval))
                time_curr = time.time()
                loss_display = 0
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
        self.cond_encoder.load_state_dict(checkpoint['encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        print('Loaded checkpoint from: {}'.format(path))
        del checkpoint


    def save_ckpt(self):
        ckpt = {'denoiser': self.denoiser.state_dict(), 'encoder':self.cond_encoder.state_dict(),
                'denoiser_ema': self.ema_denoiser.state_dict(), 'encoder_ema': self.ema_cond_encoder.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'step': self.step}
        torch.save(ckpt, os.path.join(self.cfg.save_path, f'ckpt_step_'+ str(self.step) +'.tar'))


    def update_ema(self):
        # 遍历主模型和EMA模型的参数，成对更新
        for model_param, ema_param in zip(self.denoiser.parameters(), self.ema_denoiser.parameters()):
            ema_param.data.mul_(self.cfg.ema_rate).add_(model_param.data, alpha=1.0 - self.cfg.ema_rate)
            # 确保ema_param始终处于detach状态（虽然初始化时已冻结，但再次确认）
            ema_param.detach_()

        for model_param, ema_param in zip(self.cond_encoder.parameters(), self.ema_cond_encoder.parameters()):
            ema_param.data.mul_(self.cfg.ema_rate).add_(model_param.data, alpha=1.0 - self.cfg.ema_rate)
            ema_param.detach_()
