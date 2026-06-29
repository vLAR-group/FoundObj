import argparse
import logging
import os, torch
import warnings
warnings.filterwarnings("ignore")
from omegaconf import OmegaConf
from safetensors.torch import load_file
os.environ["ATTN_BACKEND"] = "xformers"
from foundobj.datasets.scannet_cutcost import VoxelizedDataset
from foundobj.trainers.cutcost_trellis_trainer import Trainer
from mask3d_spconv.mask3d import Mask3D
from mask3d_spconv.sparse_unet import Res16UNet34C
from RLNet.RLnet import PPONet
from trellis_tools.models.cf_and_atten import CenterField, SparseStructureEncoder
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40960, rlimit[1]))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(8)
os.environ["CF_DBSCAN_JOBS"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Train FoundObj CutCost Trellis model")
    data = parser.add_argument_group("Data and outputs")
    data.add_argument("--data-root", default='data/scannet/processed_aligns')
    data.add_argument("--superpoint-dir", default='data/scannet/superpoints')
    data.add_argument("--dino-dir", default='data/scannet/dinov2b14_unscene3dspfeats')
    data.add_argument("--superpoint-neighbor-dir", default="data/scannet/superpoint_neighbors")
    data.add_argument("--pre-pseudo", default="pseudo_mask")
    data.add_argument("--save-path", default="ckpts")

    model = parser.add_argument_group("Model checkpoints")
    model.add_argument("--mask3d-config", default="mask3d_spconv/mask3d_scannet.yaml")
    model.add_argument("--sparse-encoder-ckpt", default="ss_enc_conv3d_16l8_fp16.safetensors")
    model.add_argument("--center-field-ckpt", default="centerfield_ckpt.tar")

    train = parser.add_argument_group("Training")
    train.add_argument("--batch_size", type=int, default=5)
    train.add_argument("--num_epochs", type=int, default=1000)
    train.add_argument("--num_workers", type=int, default=8)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--voxel_size", type=float, default=0.02)
    train.add_argument("--batch-iter", type=int, default=1)
    train.add_argument("--use-sp", default=True)

    rollout = parser.add_argument_group("RL rollout")
    rollout.add_argument("--block_num", type=int, default=150)
    rollout.add_argument("--init_sp_num", type=int, default=100)
    rollout.add_argument("--nbr_sp_num", type=int, default=50)
    rollout.add_argument("--env_radius", type=float, default=1)
    rollout.add_argument("--max-step", type=int, default=5)
    rollout.add_argument("--max-eval-step", type=int, default=5)
    rollout.add_argument("--trajectory-capacity", type=int, default=50)

    ppo = parser.add_argument_group("PPO")
    ppo.add_argument("--rewarder_batch_size", type=int, default=100)
    ppo.add_argument("--rl-gamma", type=float, default=0.90)
    ppo.add_argument("--clip-actor-eps", type=float, default=0.2)
    ppo.add_argument("--gae-lambda", type=float, default=0.9)
    ppo.add_argument("--gae", default=True)
    ppo.add_argument("--ent_coeff", type=float, default=1e-1)
    ppo.add_argument("--clip-value", default=False)
    ppo.add_argument("--clip-value-eps", type=float, default=0.1)
    ppo.add_argument("--normalize-adv", default=True)

    reward = parser.add_argument_group("Reward and proposal filtering")
    reward.add_argument("--success-reward", type=float, default=10)
    reward.add_argument("--iou-thr", type=float, default=0.5)
    reward.add_argument("--mask-min-voxel", type=int, default=50)
    reward.add_argument("--initsp_min_voxel", type=int, default=10)
    reward.add_argument("--initsp-max-voxel", type=int, default=10)
    reward.add_argument("--objsp-height-min", type=float, default=0.2)
    reward.add_argument("--objsp-height-max", type=float, default=1.8)
    reward.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def main(args, logger):
    mask3d = build_mask3d(args.mask3d_config)
    pponet = PPONet()

    encoder = SparseStructureEncoder(in_channels=1, latent_channels=8, num_res_blocks=2, channels=[32, 128, 512])
    encoder.load_state_dict(load_file(args.sparse_encoder_ckpt))
    decoder = CenterField(latent_channels=512, num_atten_blocks=3, num_heads=16)
    decoder.load_state_dict(torch.load(args.center_field_ckpt, weights_only=False)["decoder"])

    train_dataset = VoxelizedDataset("train", args, batch_size=args.batch_size)
    val_dataset = VoxelizedDataset("validation", args, batch_size=1)
    trainer = Trainer(mask3d, pponet, [encoder, decoder], logger, train_dataset, val_dataset, args.save_path, args)
    trainer.train_model(args.num_epochs)


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
    logger = set_logger(os.path.join(args.save_path, 'train.log'))
    os.system(f"cp {__file__} {args.save_path}")
    os.system(f"cp -r {'foundobj/'} {args.save_path}")
    os.system(f"cp -r {'mask3d_spconv/'} {args.save_path}")
    os.system(f"cp -r {'RLNet/'} {args.save_path}")
    os.system(f"cp -r {'trellis_tools/'} {args.save_path}")
    main(args, logger)
