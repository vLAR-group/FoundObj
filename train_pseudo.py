import argparse
import logging
import os, torch
import warnings
warnings.filterwarnings("ignore")
from omegaconf import OmegaConf
from foundobj.datasets.scannet_pseudo import VoxelizedDataset
from foundobj.trainers.pseudo_trainer import PseudoTrainer
from mask3d_spconv.mask3d import Mask3D
from mask3d_spconv.sparse_unet import Res16UNet34C
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40960, rlimit[1]))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mask3D with pseudo labels on ScanNet")
    parser.add_argument("--data-dir", type=str, default="data/scannet/processed_aligns")
    parser.add_argument("--sp-dir", type=str, default='data/scannet/superpoints',
                        help="Superpoint directory (None = use default sp_idx from data)")
    parser.add_argument("--pseudo-dir", type=str, default='pseudo_mask_new2',
                        help="Directory containing pseudo label .npy files")
    parser.add_argument("--save-path", type=str, default="outputs/train_pseudo")
    parser.add_argument("--mask3d-config", type=str, default="mask3d_spconv/mask3d_scannet.yaml")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=600)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--use-sp", action="store_true", default=True)
    return parser.parse_args()


def main(args, logger):
    mask3d = build_mask3d(args.mask3d_config)
    train_dataset = VoxelizedDataset('train', args, batch_size=args.batch_size)
    val_dataset = VoxelizedDataset('validation', args, batch_size=1)
    trainer = PseudoTrainer(mask3d, logger, train_dataset, val_dataset, args.save_path, args)
    trainer.train_model(args.num_epochs)


def build_mask3d(config_path):
    model_cfg = OmegaConf.load(config_path)
    backbone = Res16UNet34C(in_channels=6)
    params = {k: v for k, v in model_cfg.items() if k in Mask3D.__init__.__code__.co_varnames}
    return Mask3D(backbone, **params)


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
    return logger

if __name__ == '__main__':
    args = parse_args()

    '''Setup logger'''
    os.makedirs(args.save_path, exist_ok=True)
    logger = set_logger(os.path.join(args.save_path, 'train.log'))
    os.system(f"cp {__file__} {args.save_path}")
    os.system(f"cp -r {'./foundobj/'} {args.save_path}")
    os.system(f"cp -r {'./mask3d_spconv/'} {args.save_path}")
    main(args, logger)
