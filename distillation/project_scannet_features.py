"""Project DINOv2 image features onto ScanNet 3D superpoints for distillation."""

import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from imageio.v2 import imread
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# ──────────────────────────────────────────────────────────────────────────────
# Camera utilities
# ──────────────────────────────────────────────────────────────────────────────

def make_intrinsic(fx, fy, mx, my):
    K = np.eye(4, dtype=np.float32)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, mx, my
    return K


def adjust_intrinsic(K, src_dim, dst_dim):
    """Scale intrinsic from src_dim (W,H) to dst_dim (W,H) accounting for resize+crop."""
    if src_dim == dst_dim:
        return K
    K = K.copy()
    resize_w = int(math.floor(dst_dim[1] * src_dim[0] / src_dim[1]))
    K[0, 0] *= resize_w / src_dim[0]
    K[1, 1] *= dst_dim[1] / src_dim[1]
    K[0, 2] *= (dst_dim[0] - 1) / (src_dim[0] - 1)
    K[1, 2] *= (dst_dim[1] - 1) / (src_dim[1] - 1)
    return K


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def resize_crop(image, target_wh, interpolation=Image.NEAREST):
    """Resize preserving aspect ratio, then center-crop to exact target (W, H)."""
    w, h = image.shape[1], image.shape[0]
    if (w, h) == target_wh:
        return image
    new_h = target_wh[1]
    new_w = int(math.floor(new_h * w / h))
    image = transforms.Resize([new_h, new_w], interpolation=interpolation)(Image.fromarray(image))
    image = transforms.CenterCrop([target_wh[1], target_wh[0]])(image)
    return np.asarray(image)


def load_pose(path):
    with open(path) as f:
        lines = f.read().splitlines()
    assert len(lines) == 4, f"Bad pose file: {path}"
    return np.array([[float(v) for v in l.split()] for l in lines], dtype=np.float32)


def load_depth(path, image_dim, depth_scale):
    depth = imread(path)
    depth = resize_crop(depth, image_dim)
    return depth.astype(np.float32) / depth_scale


def load_points(scene_id, scannet_3d_root, splits):
    for split in splits:
        for name in (scene_id, scene_id.replace("scene", "")):
            path = os.path.join(scannet_3d_root, split, f"{name}_vh_clean_2.pth")
            if os.path.exists(path):
                return torch.load(path, weights_only=False, map_location="cpu")[0].astype(np.float32)
    raise FileNotFoundError(f"3D points not found for {scene_id}")


def load_superpoints(path):
    return np.load(path).astype(np.int64)


def list_scenes(superpoint_root):
    return sorted(
        os.path.basename(p).removesuffix("_sp.npy")
        for p in glob.glob(os.path.join(superpoint_root, "*_sp.npy"))
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3D-2D projection (GPU)
# ──────────────────────────────────────────────────────────────────────────────

class Projector:
    """Projects 3D points to 2D image pixels with depth-based visibility check."""

    def __init__(self, image_dim, intrinsic, vis_threshold, cut_bound, device):
        self.W, self.H = image_dim
        self.vis_threshold = vis_threshold
        self.cut_bound = cut_bound
        self.device = device
        self.K = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)

    def project(self, pose, coords_h, depth):
        """
        Args:
            pose: (4,4) camera-to-world numpy array
            coords_h: (4, N) homogeneous 3D coords on GPU
            depth: (H, W) numpy depth map
        Returns:
            (visible_idx, rows, cols) — long tensors on device
        """
        W2C = torch.as_tensor(np.linalg.inv(pose), dtype=torch.float32, device=self.device)
        cam = W2C @ coords_h
        z = cam[2]

        px = torch.round(cam[0] * self.K[0, 0] / z + self.K[0, 2]).long()
        py = torch.round(cam[1] * self.K[1, 1] / z + self.K[1, 2]).long()

        b = self.cut_bound
        inside = ((z > 1e-6)
                  & (px >= b) & (px < self.W - b)
                  & (py >= b) & (py < self.H - b))

        if not inside.any():
            e = torch.empty(0, dtype=torch.long, device=self.device)
            return e, e, e

        depth_t = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        d_proj = depth_t[py[inside], px[inside]]
        visible = (z[inside] - d_proj).abs() <= self.vis_threshold * d_proj

        idx = torch.where(inside)[0][visible]
        return idx, py[idx], px[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class FrameDataset(Dataset):
    def __init__(self, scene_dir, frame_ids, image_dim):
        self.scene_dir = scene_dir
        self.frame_ids = list(frame_ids)
        self.image_dim = image_dim
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        fid = self.frame_ids[idx]
        img = np.asarray(Image.open(os.path.join(self.scene_dir, "color", f"{fid}.jpg")).convert("RGB"))
        img = resize_crop(img, self.image_dim, interpolation=Image.BICUBIC)
        img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return self.normalize(img), fid


def collate_fn(batch):
    imgs, fids = zip(*batch)
    return torch.stack(imgs), list(fids)


# ──────────────────────────────────────────────────────────────────────────────
# DINOv2 feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(model, images):
    """Extract DINOv2 patch features → (B, C, H_patch, W_patch)."""
    tokens = model(images, is_training=True)["x_norm_patchtokens"]
    B, N, C = tokens.shape
    h, w = images.shape[-2] // 14, images.shape[-1] // 14
    return tokens.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()


# ──────────────────────────────────────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────────────────────────────────────

def process_scene(args, model, projector, scene_id, device):
    scene_dir = os.path.join(args.scannet_2d_root, scene_id)
    output_path = os.path.join(args.output_dir, f"{scene_id}.pth")
    if os.path.exists(output_path):
        print(f"skip {scene_id}")
        return

    # Load 3D data
    points = load_points(scene_id, args.scannet_3d_root, args.splits)
    sp = load_superpoints(os.path.join(args.superpoint_root, f"{scene_id}_sp.npy"))
    assert points.shape[0] == sp.shape[0], f"Point/SP mismatch: {points.shape[0]} vs {sp.shape[0]}"
    N = points.shape[0]

    # Precompute homogeneous coords on GPU
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    coords_h = torch.cat([pts, torch.ones(N, 1, device=device)], dim=1).T  # (4, N)
    del pts

    # Dataloader
    frame_ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(os.path.join(scene_dir, "color")) if f.endswith(".jpg"))
    loader = DataLoader(FrameDataset(scene_dir, frame_ids, tuple(args.model_image_dim)), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Accumulate per-point features on CPU to avoid GPU OOM
    feat_sums = None
    feat_counts = torch.zeros(N, 1)
    cam_dim = tuple(args.camera_image_dim)

    for images, fids in loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            feats = extract_features(model, images)
            feats = F.interpolate(feats.float(), size=(cam_dim[1], cam_dim[0]), mode="bicubic", align_corners=False)

        if feat_sums is None:
            feat_sums = torch.zeros(N, feats.shape[1])

        for feat, fid in zip(feats, fids):
            pose = load_pose(os.path.join(scene_dir, "pose", f"{fid}.txt"))
            depth = load_depth(os.path.join(scene_dir, "depth", f"{fid}.png"), cam_dim, args.depth_scale)
            idx, rows, cols = projector.project(pose, coords_h, depth)
            if idx.numel() == 0:
                continue
            idx_cpu = idx.cpu()
            feat_sums.index_add_(0, idx_cpu, feat[:, rows, cols].T.float().cpu())
            feat_counts.index_add_(0, idx_cpu, torch.ones(idx_cpu.numel(), 1))

    if feat_sums is None:
        raise RuntimeError(f"No features extracted for {scene_id}")

    # Aggregate to superpoints
    point_feats = feat_sums / feat_counts.clamp_min(1.0)
    observed = feat_counts.squeeze() > 0

    num_sp = int(sp[sp >= 0].max()) + 1 if (sp >= 0).any() else 0
    sp_t = torch.as_tensor(sp, dtype=torch.long)
    valid = (sp_t >= 0) & observed

    sp_sums = torch.zeros(num_sp, point_feats.shape[1])
    sp_counts = torch.zeros(num_sp, 1)
    sp_sums.index_add_(0, sp_t[valid], point_feats[valid])
    sp_counts.index_add_(0, sp_t[valid], torch.ones(int(valid.sum()), 1))
    sp_feats = sp_sums / sp_counts.clamp_min(1.0)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(sp_feats, output_path)

    if args.pca_vis:
        sp_colors = pca_rgb(sp_feats.numpy())
        pt_colors = np.full((N, 3), 128, dtype=np.uint8)
        v = (sp >= 0) & (sp < num_sp)
        pt_colors[v] = sp_colors[sp[v]]
        write_ply(os.path.join(args.output_dir, f"{scene_id}.ply"), points, pt_colors)

    print(f"saved {output_path}  shape={tuple(sp_feats.shape)}")
    del coords_h, feat_sums, feat_counts, point_feats, sp_sums, sp_counts, sp_feats


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def pca_rgb(features):
    """PCA → RGB for visualization."""
    x = features.astype(np.float32)
    valid = np.isfinite(x).all(axis=1)
    colors = np.full((len(x), 3), 128, dtype=np.uint8)
    if valid.sum() < 3:
        return colors
    xv = x[valid] - x[valid].mean(0)
    _, _, vt = np.linalg.svd(xv, full_matrices=False)
    proj = xv @ vt[:3].T
    lo, hi = proj.min(0, keepdims=True), proj.max(0, keepdims=True)
    proj = (proj - lo) / np.maximum(hi - lo, 1e-6)
    colors[valid] = (proj * 255).clip(0, 255).astype(np.uint8)
    return colors


def write_ply(path, coords, colors):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(coords)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(coords, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Project DINOv2 features to ScanNet superpoints")
    p.add_argument("--scannet-2d-root", default="/home/zihui/HDD/LogoSP_release/data/ScanNet/scannet_2d")
    p.add_argument("--scannet-3d-root", default="/home/zihui/HDD/LogoSP_release/data/ScanNet/scannet_3d")
    p.add_argument("--superpoint-root", default=os.path.join(REPO_ROOT, "data/scannet/superpoints"))
    # p.add_argument("--superpoint-root", default='/home/zihui/SSD/RLTrellis/outputs')
    p.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "data/scannet/dinov2b14_unscene3dspfeats"))
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--pca-vis", default=True)
    p.add_argument("--model", default="dinov2_vitb14_reg")
    p.add_argument("--hub-repo", default="facebookresearch/dinov2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--model-image-dim", type=int, nargs=2, default=[1260, 952], metavar=("W", "H"))
    p.add_argument("--camera-image-dim", type=int, nargs=2, default=[320, 240], metavar=("W", "H"))
    p.add_argument("--original-image-dim", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    p.add_argument("--depth-scale", type=float, default=1000.0)
    p.add_argument("--visibility-threshold", type=float, default=0.25)
    p.add_argument("--cut-bound", type=int, default=10)
    p.add_argument("--fx", type=float, default=577.870605)
    p.add_argument("--fy", type=float, default=577.870605)
    p.add_argument("--mx", type=float, default=319.5)
    p.add_argument("--my", type=float, default=239.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    K = make_intrinsic(args.fx, args.fy, args.mx, args.my)
    K = adjust_intrinsic(K, tuple(args.original_image_dim), tuple(args.camera_image_dim))
    projector = Projector(tuple(args.camera_image_dim), K, args.visibility_threshold, args.cut_bound, device)

    model = torch.hub.load(args.hub_repo, args.model).to(device).eval()

    scenes = list_scenes(args.superpoint_root)
    print(f"Processing {len(scenes)} scenes → {args.output_dir}")
    for scene_id in scenes:
        try:
            process_scene(args, model, projector, scene_id, device)
        except Exception as e:
            print(f"FAILED {scene_id}: {e}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
