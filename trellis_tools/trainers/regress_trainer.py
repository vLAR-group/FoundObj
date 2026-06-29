import torch
import torch.nn.functional as F
from models.structured_latent_distill_flow import SLatFlowModel
from torch.utils.data import DataLoader
import numpy as np
import os, copy, time
from easydict import EasyDict as edict
from glob import glob

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
        self.trainloader = DataLoader(self.trainset, batch_size=self.cfg.batch_size_per_gpu, pin_memory=True,
            num_workers=self.cfg.num_workers, drop_last=True, persistent_workers=False, shuffle=True, collate_fn=self.trainset.collate_fn)

    def training_loss(self, spconv, dino):
        pred_dino = self.denoiser(spconv['x_in'])
        pred_dino = pred_dino.feats
        terms = edict()
        terms["regress_loss"] = F.mse_loss(pred_dino, dino)
        return terms

    def run_step(self, batch):
        spconv, dino = batch ### clean voxel, aug_pc, batch data are automatically in cuda??
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = self.training_loss(spconv, dino)
            l = loss['regress_loss']
        l.backward()
        ## gradient clip
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), self.cfg.max_norm)

        ## step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_ema()
        return l.item(), loss['regress_loss'].item()

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
        for model_param, ema_param in zip(self.denoiser.parameters(), self.ema_denoiser.parameters()):
            ema_param.data.mul_(self.cfg.ema_rate).add_(model_param.data, alpha=1.0 - self.cfg.ema_rate)
            ema_param.detach_()