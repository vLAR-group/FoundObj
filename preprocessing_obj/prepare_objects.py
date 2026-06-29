"""Step 1: download/convert object meshes into data/objects/<dataset>/renders/<sha256>/mesh.ply.

Each mesh is rotated Y-up -> Z-up (the scene builder assumes Z is up) and normalized
to a unit cube at the origin. Object lists (sha256, file_identifier, aesthetic_score) are derived from
TRELLIS-500K metadata (JeffreyXiang/TRELLIS-500K) and stored under
object_lists/; only assets with aesthetic_score >= 4.5 are used.

  ABO:        downloaded automatically from the public abo-3dmodels.tar archive.
                  python prepare_objects.py --dataset ABO
  3D-Future:  licensed (https://tianchi.aliyun.com/dataset/98063) - download
              3D-FUTURE-model.zip manually, extract, and pass --models_root
              as either the raw dir containing 3D-FUTURE-model/ or that dir itself.
                  python prepare_objects.py --dataset 3D-Future --models_root /path/to/3D-FUTURE_raw
"""
import os
import argparse
import urllib.request
import tarfile
import hashlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import pandas as pd
import trimesh

OBJECT_LISTS_DIR = os.path.join(os.path.dirname(__file__), "object_lists")
ABO_TAR_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar"
Y_UP_TO_Z_UP = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])


def _remote_size(headers, fallback=0):
    content_range = headers.get("Content-Range")
    if content_range and "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    content_length = int(headers.get("Content-Length") or 0)
    return fallback + content_length if fallback else content_length


def download_with_progress(url, out_path):
    part_path = out_path + ".part"
    downloaded = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    request = urllib.request.Request(url)
    if downloaded:
        request.add_header("Range", f"bytes={downloaded}-")

    with urllib.request.urlopen(request) as response:
        total = _remote_size(response.headers, downloaded)
        mode = "ab" if downloaded else "wb"
        with open(part_path, mode) as f, tqdm(
            total=total or None, initial=downloaded, unit="B", unit_scale=True,
            unit_divisor=1024, desc="Downloading ABO"
        ) as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

    final_size = os.path.getsize(part_path)
    if total and final_size != total:
        raise RuntimeError(
            f"ABO archive download incomplete: {part_path} ({final_size}/{total} bytes). Rerun to resume."
        )
    os.replace(part_path, out_path)


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_mesh(mesh, out_path):
    """Y-up -> Z-up, normalize to a unit cube at the origin, and write mesh.ply."""
    mesh.apply_transform(Y_UP_TO_Z_UP)
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2)
    mesh.apply_scale(1.0 / (hi - lo).max())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mesh.export(out_path)


def resolve_3dfuture_obj(models_root, file_identifier):
    """Resolve a 3D-FUTURE raw_model.obj from either extracted-root convention."""
    rel_path = os.path.join(file_identifier, "raw_model.obj")
    candidates = [os.path.join(models_root, rel_path)]
    parts = file_identifier.split("/", 1)
    if len(parts) == 2 and os.path.basename(os.path.normpath(models_root)) == parts[0]:
        candidates.append(os.path.join(models_root, parts[1], "raw_model.obj"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def ensure_abo_archive(out_dir):
    raw_dir = os.path.join(out_dir, "raw")
    tar_path = os.path.join(raw_dir, "abo-3dmodels.tar")
    os.makedirs(raw_dir, exist_ok=True)
    if not os.path.exists(tar_path):
        print(f"Downloading ABO archive to {tar_path}")
        download_with_progress(ABO_TAR_URL, tar_path)
    return tar_path


def extract_abo_sources(rows, raw_dir, tar_path):
    requested = {}
    for row in rows:
        src_path = os.path.join(raw_dir, "3dmodels", "original", row.file_identifier)
        if not os.path.exists(src_path):
            requested[f"3dmodels/original/{row.file_identifier}"] = row.file_identifier
    if not requested:
        return

    found = set()
    try:
        with tarfile.open(tar_path, "r:") as tar:
            with tqdm(total=len(requested), desc="Extracting ABO", unit="obj") as pbar:
                for member in tar:
                    if member.name in requested:
                        tar.extract(member, path=raw_dir)
                        found.add(member.name)
                        pbar.update(1)
                        if len(found) == len(requested):
                            break
    except tarfile.ReadError as exc:
        raise RuntimeError(
            f"ABO archive is incomplete or corrupt: {tar_path}. Move it to {tar_path}.part to resume, or replace it manually."
        ) from exc

    for member in sorted(set(requested) - found):
        print(f"missing in ABO archive: {member}")


def prepare_abo_objects(rows, out_dir):
    missing_rows = [
        row for row in rows
        if not os.path.exists(os.path.join(out_dir, "renders", row.sha256, "mesh.ply"))
    ]
    if not missing_rows:
        return

    raw_dir = os.path.join(out_dir, "raw")
    tar_path = ensure_abo_archive(out_dir)
    extract_abo_sources(missing_rows, raw_dir, tar_path)

    for row in tqdm(missing_rows, desc="Converting ABO", unit="obj"):
        out_path = os.path.join(out_dir, "renders", row.sha256, "mesh.ply")
        if os.path.exists(out_path):
            continue
        src_path = os.path.join(raw_dir, "3dmodels", "original", row.file_identifier)
        if not os.path.exists(src_path):
            continue
        try:
            if file_sha256(src_path) != row.sha256:
                print(f"sha256 mismatch: {row.file_identifier}")
                continue
            mesh = trimesh.load(src_path, force="mesh")
            export_mesh(mesh, out_path)
        except Exception as exc:
            print(f"failed: {row.file_identifier}: {exc}")


def prepare_one(row, out_dir, models_root):
    out_path = os.path.join(out_dir, "renders", row.sha256, "mesh.ply")
    if os.path.exists(out_path):
        return
    try:
        obj_path = resolve_3dfuture_obj(models_root, row.file_identifier)
        if not os.path.isfile(obj_path):
            print(f"missing: {obj_path}")
            return
        mesh = trimesh.load(obj_path, force="mesh")
        export_mesh(mesh, out_path)
    except Exception as exc:
        print(f"failed: {row.file_identifier}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Download/convert object meshes (ABO or 3D-Future)")
    parser.add_argument("--dataset", default="ABO", choices=["ABO", "3D-Future"])
    parser.add_argument("--output_dir", default=None, help="default: data/objects/<dataset>")
    parser.add_argument("--models_root", default=None,
                        help="3D-Future only: extracted raw dir containing 3D-FUTURE-model/ or that folder itself")
    args = parser.parse_args()
    if args.dataset == "3D-Future" and args.models_root is None:
        parser.error("3D-Future needs --models_root (extracted 3D-FUTURE-model.zip from "
                     "https://tianchi.aliyun.com/dataset/98063)")

    out_dir = args.output_dir or os.path.join("data/objects", args.dataset)
    df = pd.read_csv(os.path.join(OBJECT_LISTS_DIR, f"{args.dataset}.csv"))
    df = df[df.aesthetic_score >= 4.5]
    rows = [row for _, row in df.iterrows()]
    print(f"[{args.dataset}] {len(rows)} objects -> {out_dir}/renders/<sha256>/mesh.ply")
    if args.dataset == "ABO":
        prepare_abo_objects(rows, out_dir)
    else:
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda row: prepare_one(row, out_dir, args.models_root), rows))


if __name__ == "__main__":
    main()
