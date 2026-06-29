import torch
import torch.nn.functional as F
from models.sparse_structure_vae import SparseStructureEncoder
from models.sparse_structure_flow_spkv import SparseStructureFlowModel
from sparse_unet.spconv_unet_v1m2_layernorm import Res16UNet18
from datasets.point_spconv import PointSpConv
from torch.utils.data import DataLoader
import numpy as np
import os, copy, time
from easydict import EasyDict as edict
from glob import glob

from safetensors.torch import load_file
import spconv.pytorch as spconv

# def cycle(dataloader: DataLoader):
#     """将DataLoader包装成无限循环迭代器，自动处理epoch切换"""
#     while True:
#         for batch in dataloader:
#             yield batch
#         # 处理分布式采样器的epoch更新（确保每个epoch打乱不同）
#         if hasattr(dataloader.sampler, 'set_epoch'):
#             print("Setting new epoch for dataloader sampler.")
#             current_epoch = getattr(dataloader.sampler, 'epoch', 0)
#             dataloader.sampler.set_epoch(current_epoch + 1)
#
# def recursive_to_device(data, device: torch.device, non_blocking: bool = False):
#     """
#     Recursively move all tensors in a data structure to a device.
#     """
#     if hasattr(data, "to"):
#         return data.to(device, non_blocking=non_blocking)
#     elif isinstance(data, (list, tuple)):
#         return type(data)(recursive_to_device(d, device, non_blocking) for d in data)
#     elif isinstance(data, dict):
#         return {k: recursive_to_device(v, device, non_blocking) for k, v in data.items()}
#     else:
#         return data

class Trainer:
    def __init__(self, cfg, logger, prefetch_data=True):
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

        self.cond_encoder = Res16UNet18(in_channels=1).to(self.device)
        self.denoiser = SparseStructureFlowModel(**self.cfg.models.denoiser.args).to(self.device)
        ## optimizer
        self.optimizer = torch.optim.AdamW([{'params': self.cond_encoder.parameters()},
                                            {'params': self.denoiser.parameters()}], lr=self.cfg.lr)
        ## EMA
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.ema_cond_encoder = copy.deepcopy(self.cond_encoder)
        for param in self.ema_denoiser.parameters():
            param.requires_grad = False
        for param in self.ema_cond_encoder.parameters():
            param.requires_grad = False

    def prepare_dataloader(self):
        self.trainset = PointSpConv(self.cfg.data_dir, min_aesthetic_score=4.5)
        self.trainloader = DataLoader(self.trainset, batch_size=self.cfg.batch_size_per_gpu, pin_memory=True, shuffle=True,
            num_workers=self.cfg.workers, drop_last=True, persistent_workers=True, collate_fn=self.trainset.collate_fn)
        # self.data_iterator = cycle(self.trainloader)

    # def load_data(self):
    #     if self.prefetch_data:
    #         if self._data_prefetched is None:
    #             self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
    #         data = self._data_prefetched
    #         self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
    #     else:
    #         data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
    #     return data

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

    def training_loss(self, x_1: torch.Tensor, cond_grids, cond_feats):
        noise = torch.randn_like(x_1)  # noise is x_0, torch.Size([8, 8, 16, 16, 16])
        t = self.sample_t(x_1.shape[0]).to(x_1.device).float()
        x_t = self.diffuse(x_1, t, noise=noise)  # torch.Size([8, 8, 16, 16, 16])
        in_field = spconv.SparseConvTensor(features=cond_feats.cuda(), indices=cond_grids.int().cuda(),
                    spatial_shape=list(cond_grids.max(0)[0] + 16)[1:], batch_size=cond_grids.max(0)[0][0].item() + 1)

        cond, fullres = self.cond_encoder(in_field)# SparseTensor [K, C]
        ## complete dense voxel
        batch_grids, batch_feats, kv_seqlen = [], [], []
        for b in range(x_1.shape[0]): ## here, I'm not sure whether unique is necessary?
            bs_mask = torch.where(cond.indices[:, 0] == b)[0]
            bs_grid, bs_feats = cond.indices[:, 1:][bs_mask], cond.features[bs_mask]
            batch_grids.append(bs_grid), batch_feats.append(bs_feats)
            kv_seqlen.append(bs_feats.shape[0])
        cond_feats, cond_grids = torch.cat(batch_feats), torch.cat(batch_grids)
        cond_grids = cond_grids.type(cond_feats.dtype)
        ######################################################################################################
        pred = self.denoiser(x_t, t * 1000, [cond_feats, cond_grids], kv_seqlen)
        assert pred.shape == noise.shape == x_1.shape
        target = self.get_v(x_1, noise)
        terms = edict()
        terms["gen_loss"] = F.mse_loss(pred, target)
        return terms

    def run_step(self, batch):
        x_1, cond_grids, cond_feats = batch ### clean voxel, aug_pc, batch data are automatically in cuda??
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                grids = x_1.float().to(self.device)
                z = self.supervision_encoder(grids, sample_posterior=False)  # [B, 1, 64, 64, 64]
            ## training
            loss = self.training_loss(z, cond_grids, cond_feats)
            l = loss['gen_loss']
        ## backward
        l.backward()

        ## gradient clip
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), self.cfg.max_norm)

        ## step
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Update exponential moving average
        self.update_ema()
        return l.item(), loss['gen_loss'].item()

    def run(self):
        loss_display, gen_loss_display = 0, 0
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
            # batch = self.load_data()
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
