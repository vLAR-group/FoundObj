import argparse
import logging
import os, torch
import warnings
warnings.filterwarnings("ignore")
from omegaconf import OmegaConf
from foundobj.datasets.scannet_cutcost import VoxelizedDataset
from foundobj.trainers.eval_scannet_trainer import EvalScanNetTrainer
from mask3d_spconv.mask3d import Mask3D
from mask3d_spconv.sparse_unet import Res16UNet34C
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40960, rlimit[1]))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(8)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on ScanNet200")
    parser.add_argument("--data-root", type=str, default="data/scannet/processed_aligns200")
    parser.add_argument("--superpoint-dir", type=str, default='data/scannet/superpoints')
    parser.add_argument("--save-path", type=str, default="outputs/eval_scannet")
    parser.add_argument("--ckpt", type=str, default='ckpts/reproduce_pseudo_again/checkpoint_360.tar')
    parser.add_argument("--mask3d-config", type=str, default="mask3d_spconv/mask3d_scannet.yaml")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--use-sp", action="store_true", default=True)
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--mask-min-voxel", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main(args, logger):
    mask3d = build_mask3d(args.mask3d_config)
    val_dataset = VoxelizedDataset("validation", args, batch_size=1)
    trainer = EvalScanNetTrainer(mask3d, logger, val_dataset, args.save_path, args)
    trainer.validation(vis=args.vis, log=False, ckpt_path=args.ckpt)


def build_mask3d(config_path):
    model_cfg = OmegaConf.load(config_path)
    backbone = Res16UNet34C(in_channels=6)
    params = {k: v for k, v in model_cfg.items() if k in Mask3D.__init__.__code__.co_varnames}
    return Mask3D(backbone, **params)


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
    return logger

if __name__ == '__main__':
    args = parse_args()

    '''Setup logger'''
    os.makedirs(args.save_path, exist_ok=True)
    logger = set_logger(os.path.join(args.save_path, 'eval.log'))
    main(args, logger)
