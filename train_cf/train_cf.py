import logging
import argparse
import os
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if p not in {REPO_ROOT, SCRIPT_DIR}]
sys.path.insert(0, REPO_ROOT)

from easydict import EasyDict as edict
os.environ['ATTN_BACKEND'] = 'flash_attn'
import warnings
warnings.filterwarnings("ignore")
from train_cf.trainers.cf_trainer import Trainer


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/objects/ABO')
    parser.add_argument("--save_path", type=str, default='ckpt/centerfield')
    parser.add_argument("--batch_size_per_gpu", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl", type=float, default=1e-4)
    parser.add_argument("--ema_rate", type=float, default=0.9999)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    opt = parser.parse_args()
    cfg = edict()
    cfg.models = edict({
        "encoder": edict({
            "name": "SparseStructureEncoder",
            "args": edict({
                "in_channels": 1,
                "latent_channels": 8,
                "num_res_blocks": 2,
                "num_res_blocks_middle": 2,
                "channels": [32, 128, 512],
            }),
        }),
    })
    cfg.update(opt.__dict__)
    return cfg


def main(cfg, logger):
    trainer = Trainer(cfg, logger)
    trainer.run()


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
    cfg = config_parser()

    os.makedirs(os.path.join(cfg.save_path), exist_ok=True)
    logger = set_logger(os.path.join(cfg.save_path, 'train.log'))

    os.system(f"cp {__file__} {os.path.join(cfg.save_path)}")
    os.system(f"cp -r {os.path.join(REPO_ROOT, 'train_cf')} {os.path.join(cfg.save_path)}")

    main(cfg, logger)
