import torch
import torch.nn.functional as F
from trellis_tools.models.cf_and_atten import SparseStructureEncoder, CenterField
from train_cf.datasets.voxel2cf import Voxel2vector
from torch.utils.data import DataLoader
import numpy as np
import os, copy, time
from easydict import EasyDict as edict
from glob import glob
from safetensors.torch import load_file
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch
        if hasattr(dataloader.sampler, 'set_epoch'):
            print("Setting new epoch for dataloader sampler.")
            current_epoch = getattr(dataloader.sampler, 'epoch', 0)
            dataloader.sampler.set_epoch(current_epoch + 1)

def recursive_to_device(data, device: torch.device, non_blocking: bool = False):
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
        self.encoder = SparseStructureEncoder(**self.cfg.models.encoder.args).to(self.device)
        self.encoder.load_state_dict(load_file(os.path.join(REPO_ROOT, "ss_enc_conv3d_16l8_fp16.safetensors")))

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.decoder = CenterField(latent_channels=512, num_atten_blocks=3, num_heads=16).to(self.device)
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.cfg.lr, weight_decay=0.0)
        self.ema_decoder = copy.deepcopy(self.decoder)
        for param in self.ema_decoder.parameters():
            param.requires_grad = False


    def prepare_dataloader(self):
        self.trainset = Voxel2vector(self.cfg.data_dir)
        self.trainloader = DataLoader(self.trainset, batch_size=self.cfg.batch_size_per_gpu, pin_memory=True, collate_fn=self.trainset.collate_fn,
                            num_workers = self.cfg.num_workers, drop_last=True, persistent_workers=self.cfg.num_workers > 0, shuffle=True)
        self.data_iterator = cycle(self.trainloader)

    def load_data(self):
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        return data

    def training_loss(self, ss, pts, cf, q_lenseq, mask):
        with torch.no_grad():
            z = self.encoder(ss.float()).detach()
        pred_cf = self.decoder(pts, z, q_lenseq) # [N, 3]
        if torch.isnan((pred_cf - cf).pow(2)).any():
            print(f"[NaN] feat")
        if torch.isinf((pred_cf - cf).pow(2)).any():
            print(f"[Inf] feat")
        loss = torch.sum((pred_cf - cf).pow(2), dim=-1).sqrt()
        loss = (loss * mask).sum() / mask.sum().clamp_min(1) + (loss* (1 - mask)).sum() / (1 - mask).sum().clamp_min(1)
        return loss

    def run_step(self, batch):
        ss, pts, cf, q_lenseq, mask = batch
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = self.training_loss(ss, pts, cf, q_lenseq, mask)
        loss.backward()

        for n, p in self.decoder.named_parameters():
            if p.grad is None:
                print(f"[None] grad in decoder.{n}")
            if torch.isnan(p.grad).any() :
                print(f"[NaN] grad in decoder.{n}")
                break
            if torch.isinf(p.grad).any() :
                print(f"[Inf] grad in decoder.{n}")

        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.cfg.max_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_ema()
        return loss.item()

    def run(self):
        loss_display = 0
        time_curr = time.time()
        self.decoder.train()

        while self.step < self.cfg.max_steps:
            batch = self.load_data()
            loss = self.run_step(batch)
            loss_display += loss
            self.step += 1
            if self.step % self.cfg.log_interval == 0:
                loss_display /= self.cfg.log_interval
                torch.cuda.synchronize()
                time_used = time.time() - time_curr
                self.logger.info(
                    'Train Iteration: {}/{}, Loss: {:.5f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                    self.step, self.cfg.max_steps, loss_display, self.optimizer.param_groups[0]['lr'], time_used,
                    self.cfg.log_interval))
                time_curr = time.time()
                loss_display = 0
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
