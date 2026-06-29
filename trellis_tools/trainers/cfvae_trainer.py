import torch
import torch.nn.functional as F
from models.sparse_structure_vae_cf import SparseStructureEncoder, SparseStructureDecoder
from datasets.voxel2vector import Voxel2vector
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
            print("Setting new epoch for dataloader sampler.")
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
    def __init__(self, cfg, logger, prefetch_data=True):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device('cuda')
        self.step = 0
        self.init_models_optimizer_lr_scheduler()
        self.prepare_dataloader()
        self.load_ckpt()
        self.prefetch_data = prefetch_data
        if self.prefetch_data:
            self._data_prefetched = None

    def init_models_optimizer_lr_scheduler(self):
        ## models
        self.encoder = SparseStructureEncoder(**self.cfg.models.encoder.args).to(self.device)
        self.encoder.load_state_dict(load_file("ss_enc_conv3d_16l8_fp16.safetensors"))

        #设置为评估模式
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.decoder = SparseStructureDecoder(**self.cfg.models.decoder.args).to(self.device)
        missing, unexpected = self.decoder.load_state_dict(load_file("ss_dec_conv3d_16l8_fp16.safetensors"), strict=False)
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)
        print(f"[ckpt] decoder load_state_dict done (missing {len(missing)}, unexpected {len(unexpected)})")
        ## optimizer
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.cfg.lr, weight_decay=0.0)
        self.ema_decoder = copy.deepcopy(self.decoder)
        for param in self.ema_decoder.parameters():
            param.requires_grad = False


    def prepare_dataloader(self):
        self.trainset = Voxel2vector(self.cfg.data_dir, min_aesthetic_score=4.5)
        self.trainloader = DataLoader(self.trainset, batch_size=self.cfg.batch_size_per_gpu, pin_memory=True,
                            num_workers = self.cfg.num_workers, drop_last=True, persistent_workers=True, shuffle=True)
        self.data_iterator = cycle(self.trainloader)
        
    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        return data

    def vector_l2_loss(self, pred_vectors, target_vectors, mask):
        #对输出进行缩放，避免数值过大
        scale = 64.0
        pred_vectors = pred_vectors.float() / scale
        target_vectors = target_vectors.float() / scale
        mask = mask.float()
        
        diff = pred_vectors - target_vectors
        # 计算每个位置的向量长度
        #l2_distances = torch.sqrt(torch.sum(diff.float() ** 2, dim=1)).float()  # [B, H, W, D] #diff为float后是否nan
        l2_distances = torch.sum(diff.float() ** 2, dim=1).float() #不开方debug,看是否nan
        l2_distances = l2_distances * mask  # 只计算稀疏结构内的距离
        
        # 先对每个batch sample的空间维度求平均，再对batch维度求平均
        l2_distances = torch.sum(l2_distances, dim=(1, 2, 3))  # [B]
        mask = torch.sum(mask, dim=(1, 2, 3)).clamp_min(1)  # [B]
        return torch.mean(l2_distances / mask)
    
    def training_loss(self, ss, ss_vector, ss_obj):
        with torch.no_grad():
            z = self.encoder(ss.float(), sample_posterior=False, return_raw=False)
        pred_vector, pred_ss = self.decoder(z)
        terms = edict()
        mask_all = ss.squeeze(1).float()  # [B, H, W, D]
        mask_obj = ss_obj.squeeze(1).float()  # [B, H, W, D]
        mask_bg  = mask_all * (1.0 - mask_obj)
        terms['l2_obj'] = self.vector_l2_loss(pred_vector, ss_vector, mask_obj)
        terms['l2_bg'] = self.vector_l2_loss(pred_vector, ss_vector, mask_bg)
        logits = F.sigmoid(pred_ss)
        terms["dice"] = 1 - (2 * (logits * ss.float()).sum() + 1) / (logits.sum() + ss.float().sum() + 1)
        terms["loss"] = terms["l2_obj"] + terms["l2_bg"] + terms["dice"]
        return terms

    def run_step(self, batch):
        ss, ss_vector, ss_obj = batch
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            ## training
            loss = self.training_loss(ss, ss_vector, ss_obj)
            l = loss['loss']
        ## backward
        l.backward()

#########################debug grad nan inf
        for n, p in self.decoder.named_parameters():
            if p.grad is None:
                print(f"[None] grad in decoder.{n}")
            if torch.isnan(p.grad).any() :
                print(f"[NaN] grad in decoder.{n}")
                break
            if torch.isinf(p.grad).any() :
                print(f"[Inf] grad in decoder.{n}")

        # ## gradient clip
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.cfg.max_norm)

        ## step
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Update exponential moving average
        self.update_ema()
        return loss['loss'].item(), loss['l2_obj'].item(), loss['l2_bg'].item(), loss['dice'].item()

    def run(self):
        loss_display, l2_obj_loss_display, l2_bg_loss_display, dice_loss_display = 0, 0, 0, 0
        time_curr = time.time()
        self.decoder.train()

        while self.step < self.cfg.max_steps:
            batch = self.load_data()
            loss, l2_obj_loss, l2_bg_loss, dice_loss  = self.run_step(batch)
            loss_display += loss
            l2_obj_loss_display += l2_obj_loss
            l2_bg_loss_display += l2_bg_loss
            dice_loss_display += dice_loss
            self.step += 1
            ## logging
            if self.step % self.cfg.log_interval == 0:
                loss_display /= self.cfg.log_interval
                l2_obj_loss_display /= self.cfg.log_interval
                l2_bg_loss_display /= self.cfg.log_interval
                dice_loss_display /= self.cfg.log_interval
                torch.cuda.synchronize()
                time_used = time.time() - time_curr
                self.logger.info(
                    'Train Iteration: {}/{}, Loss: {:.5f}, L2 Obj Loss: {:.5f}, L2 BG Loss: {:.5f}, Dice Loss: {:.5f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                    self.step, self.cfg.max_steps, loss_display, l2_obj_loss_display, l2_bg_loss_display, dice_loss_display, self.optimizer.param_groups[0]['lr'], time_used,
                    self.cfg.log_interval))
                time_curr = time.time()
                loss_display, l2_obj_loss_display, l2_bg_loss_display, dice_loss_display = 0, 0, 0, 0
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
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        print('Loaded checkpoint from: {}'.format(path))
        del checkpoint


    def save_ckpt(self):
        ckpt = {'decoder': self.decoder.state_dict(), 'decoder_ema': self.ema_decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'step': self.step}
        torch.save(ckpt, os.path.join(self.cfg.save_path, f'ckpt_step_'+ str(self.step) +'.tar'))


    def update_ema(self):
        for model_param, ema_param in zip(self.decoder.parameters(), self.ema_decoder.parameters()):
            ema_param.data.mul_(self.cfg.ema_rate).add_(model_param.data, alpha=1.0 - self.cfg.ema_rate)
            ema_param.detach_()
