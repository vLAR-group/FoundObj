import torch
import torch.nn.functional as F
from models.sparse_structure_vae_v2 import SparseStructureEncoder, SparseStructureDecoder
from datasets.voxel2sdf import Voxel2SDF
from torch.utils.data import DataLoader
import numpy as np
import os, copy, time
from easydict import EasyDict as edict
from glob import glob

from safetensors.torch import load_file

def cycle(dataloader: DataLoader):
    """将DataLoader包装成无限循环迭代器，自动处理epoch切换"""
    while True:
        for batch in dataloader:
            yield batch
        # 处理分布式采样器的epoch更新（确保每个epoch打乱不同）
        if hasattr(dataloader.sampler, 'set_epoch'):
            current_epoch = getattr(dataloader.sampler, 'epoch', 0)
            dataloader.sampler.set_epoch(current_epoch + 1)

def recursive_to_device(data, device: torch.device, non_blocking: bool = False):
    """
    Recursively move all tensors in a data structure to a device.
    """
    if hasattr(data, "to"):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_device(d, device, non_blocking) for d in data)
    elif isinstance(data, dict):
        return {k: recursive_to_device(v, device, non_blocking) for k, v in data.items()}
    else:
        return data


class Trainer:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device('cuda')
        self.step = 0
        self.init_models_optimizer_lr_scheduler()
        self.prepare_dataloader()
        self.load_ckpt()


    def init_models_optimizer_lr_scheduler(self):
        ## models
        self.encoder = SparseStructureEncoder(**self.cfg.models.encoder.args).to(self.device)
        self.encoder.load_state_dict(load_file("ss_enc_conv3d_16l8_fp16.safetensors"))

        self.decoder = SparseStructureDecoder(**self.cfg.models.decoder.args).to(self.device)
        # self.decoder.load_state_dict(load_file("ss_dec_conv3d_16l8_fp16.safetensors"), strict=False)
        ## optimizer
        self.optimizer = torch.optim.AdamW([{'params': self.encoder.parameters()},
                                            {'params': self.decoder.parameters()}], lr=self.cfg.lr, weight_decay=0.0)
        self.ema_encoder = copy.deepcopy(self.encoder)
        for param in self.ema_encoder.parameters():
                param.requires_grad = False
        self.ema_decoder = copy.deepcopy(self.decoder)
        for param in self.ema_decoder.parameters():
                param.requires_grad = False


    def prepare_dataloader(self):
        self.trainset = Voxel2SDF(self.cfg.data_dir, min_aesthetic_score=4.5, sample_num=self.cfg.sample_num)
        self.trainloader = DataLoader(self.trainset, batch_size=self.cfg.batch_size_per_gpu, pin_memory=True,
                                num_workers = self.cfg.num_workers, drop_last=True, persistent_workers=True, shuffle=True)

    def training_loss(self, voxel, query, sdf):
        z, mean, logvar = self.encoder(voxel.float(), sample_posterior=True, return_raw=True)
        pred_sdf = self.decoder(z, query).squeeze(-1)
        terms = edict()
        # sdf_error = torch.abs(pred_sdf - sdf)
        # sdf_near_mask = (sdf_error < 0.1).float().detach()
        # sdf_w_error = sdf_error * sdf_near_mask + sdf_error * (1.0 - sdf_near_mask) * 0.5
        # terms["l1"] = sdf_w_error.mean()#F.l1_loss(pred_sdf, sdf, reduction='mean')
        terms["l1"] = F.l1_loss(pred_sdf, sdf, reduction='mean')
        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["l1"] + self.cfg.kl * terms["kl"]
        return terms


    def run_step(self, batch):
        voxel, query, sdf = batch
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            ## training
            loss = self.training_loss(voxel, query, sdf)
            l = loss['loss']
        ## backward
        l.backward()

        ## gradient clip
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.cfg.max_norm)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.cfg.max_norm)

        ## step
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Update exponential moving average
        self.update_ema()
        return loss['loss'].item(), loss['l1'].item(), loss['kl'].item()

    # def _prefetch_next_batch(self):
    #     """后台线程预取下一批数据"""
    #     try:
    #         batch = next(self.trainloader)
    #         # 异步将数据移动到设备（与计算并行）
    #         batch = [x.to(self.device, non_blocking=True) for x in batch]
    #     except Exception as e:
    #         self.logger.error(f"Prefetch error: {str(e)}")
    #         batch = None
    #
    #     with self.lock:
    #         self.next_batch = batch

    def run(self):
        loss_display, rec_loss_display, kl_display = 0, 0, 0
        time_curr = time.time()
        data_iter = iter(self.trainloader)
        self.encoder.train(), self.decoder.train()
        # for epoch in range(self.cfg.max_steps//self.cfg.batch_size_per_gpu+1):
        #     for batch_idx, batch in enumerate(self.trainloader):
        while self.step < self.cfg.max_steps:
            try:
                # 获取批次数据，如果用完则重新创建迭代器
                time_curr = time.time()
                batch = next(data_iter)
                print(time.time()-time_curr)
            except StopIteration:
                # start = time.time()
                data_iter = iter(self.trainloader)
                batch = next(data_iter)
            # batch = self.next_batch
            # loss, rec_loss, kl = self.run_step(batch)
            # loss_display += loss
            # rec_loss_display += rec_loss
            # kl_display += kl
            self.step += 1
            ## logging
            if self.step % self.cfg.log_interval == 0:
                loss_display /= self.cfg.log_interval
                rec_loss_display /= self.cfg.log_interval
                kl_display /= self.cfg.log_interval
                torch.cuda.synchronize()
                time_used = time.time() - time_curr
                self.logger.info(
                    'Train Iteration: {}/{}, Loss: {:.5f}, Rec Loss: {:.5f}, KL: {:.5f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                    self.step, self.cfg.max_steps, loss_display, rec_loss_display, kl_display, self.optimizer.param_groups[0]['lr'], time_used,
                    self.cfg.log_interval))
                time_curr = time.time()
                loss_display, rec_loss_display, kl_display = 0, 0, 0
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
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        print('Loaded checkpoint from: {}'.format(path))
        del checkpoint


    def save_ckpt(self):
        ckpt = {'encoder':self.encoder.state_dict(),'encoder_ema': self.ema_encoder.state_dict(),
                'decoder': self.decoder.state_dict(), 'decoder_ema': self.ema_decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'step': self.step}
        torch.save(ckpt, os.path.join(self.cfg.save_path, f'ckpt_step_'+ str(self.step) +'.tar'))


    def update_ema(self):
        for model_param, ema_param in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            ema_param.data.mul_(self.cfg.ema_rate).add_(model_param.data, alpha=1.0 - self.cfg.ema_rate)
            ema_param.detach_()
        for model_param, ema_param in zip(self.decoder.parameters(), self.ema_decoder.parameters()):
            ema_param.data.mul_(self.cfg.ema_rate).add_(model_param.data, alpha=1.0 - self.cfg.ema_rate)
            ema_param.detach_()
