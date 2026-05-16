#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT on camera-only masked RGB frames and export dense PLY/GLB artifacts."
    )
    parser.add_argument("--source", type=Path, default=Path("outputs/camera_only_reconstruction/masked_frames"))
    parser.add_argument("--out", type=Path, default=Path("outputs/camera_only_reconstruction_vggt"))
    parser.add_argument("--vggt-root", type=Path, default=Path("third_party/vggt"))
    parser.add_argument("--max-images", type=int, default=116)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--conf-percentile", type=float, default=55.0)
    parser.add_argument("--trim-percentile", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def find_images(source: Path, max_images: int, stride: int) -> list[Path]:
    candidates = sorted(
        p
        for p in source.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()
    )
    if stride > 1:
        candidates = candidates[::stride]
    if max_images > 0:
        candidates = candidates[:max_images]
    if not candidates:
        raise FileNotFoundError(f"No RGB images found under {source}")
    return candidates


def prepare_images(paths: list[Path], source: Path, out: Path) -> list[dict]:
    if out.exists():
        shutil.rmtree(out)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, path in enumerate(paths):
        rel = path.relative_to(source)
        safe_name = "__".join(rel.parts)
        dst = image_dir / f"{idx:04d}__{safe_name}"
        shutil.copy2(path, dst)
        manifest.append(
            {
                "index": idx,
                "source": str(path),
                "source_relative": str(rel),
                "copied": str(dst.relative_to(out)),
            }
        )
    return manifest


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().float().numpy().squeeze(0)


def filter_points(
    world_points: np.ndarray,
    depth_conf: np.ndarray,
    colors_chw: np.ndarray,
    conf_percentile: float,
    trim_percentile: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    colors = np.transpose(colors_chw, (0, 2, 3, 1))
    colors = np.clip(colors.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)
    points = world_points.reshape(-1, 3).astype(np.float32)
    conf = depth_conf.reshape(-1).astype(np.float32)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 1e-5)
    valid &= colors.sum(axis=1) >= 16

    valid_conf = conf[valid]
    conf_threshold = float(np.percentile(valid_conf, conf_percentile)) if valid_conf.size else float("inf")
    valid &= conf >= conf_threshold

    if trim_percentile > 0 and valid.sum() > 100:
        candidate_points = points[valid]
        lo = np.percentile(candidate_points, trim_percentile, axis=0)
        hi = np.percentile(candidate_points, 100.0 - trim_percentile, axis=0)
        valid &= np.all((points >= lo) & (points <= hi), axis=1)

    selected_idx = np.flatnonzero(valid)
    before_sampling = int(selected_idx.size)
    if max_points > 0 and before_sampling > max_points:
        rng = np.random.default_rng(seed)
        selected_idx = np.sort(rng.choice(selected_idx, max_points, replace=False))

    selected_points = points[selected_idx]
    selected_colors = colors[selected_idx]
    summary = {
        "raw_point_count": int(points.shape[0]),
        "valid_point_count_before_sampling": before_sampling,
        "exported_point_count": int(selected_points.shape[0]),
        "confidence_percentile": float(conf_percentile),
        "confidence_threshold": conf_threshold,
        "trim_percentile": float(trim_percentile),
        "max_points": int(max_points),
    }
    if selected_points.size:
        summary["bbox_min"] = [float(x) for x in selected_points.min(axis=0)]
        summary["bbox_max"] = [float(x) for x in selected_points.max(axis=0)]
    return selected_points, selected_colors, summary


def write_preview(points: np.ndarray, colors: np.ndarray, path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(points) == 0:
        return
    rng = np.random.default_rng(seed)
    sample_count = min(60_000, len(points))
    idx = rng.choice(len(points), sample_count, replace=False)
    pts = points[idx]
    rgb = colors[idx] / 255.0

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=rgb, s=0.35, linewidths=0)
    center = pts.mean(axis=0)
    radius = max(float(np.percentile(np.linalg.norm(pts - center, axis=1), 95)), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(out: Path, summary: dict, image_manifest: list[dict], elapsed: float) -> None:
    report = out / "quality_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VGGT Camera-only Backend Report",
        "",
        f"- Status: {summary['status']}",
        "- Backend: VGGT depth/camera feed-forward inference",
        f"- Input images: {summary['input_image_count']}",
        f"- Exported points: {summary['exported_point_count']}",
        f"- Raw model points: {summary['raw_point_count']}",
        f"- Confidence percentile: {summary['confidence_percentile']}",
        f"- Confidence threshold: {summary['confidence_threshold']}",
        f"- Runtime seconds: {elapsed:.2f}",
        "- LiDAR used: false",
        "- Forbidden inputs: lidar01, viewer_assets/*.glb, historical LiDAR-derived PLY/GLB were not read.",
        "",
        "## Outputs",
        "",
        "- `points.ply`",
        "- `scene.glb`",
        "- `preview_point_cloud.png`",
        "- `summary.json`",
        "- `images_manifest.json`",
        "",
        "## Input Evidence",
        "",
        "All images were copied from `outputs/camera_only_reconstruction/masked_frames`, which was generated from RGB camera frames only.",
        "First copied frame:",
        f"- `{image_manifest[0]['source_relative']}`",
    ]
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    vggt_root = args.vggt_root.resolve()
    if not vggt_root.exists():
        raise FileNotFoundError(f"VGGT root does not exist: {vggt_root}")
    sys.path.insert(0, str(vggt_root))

    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    start = time.time()
    source = args.source.resolve()
    out = args.out.resolve()
    image_paths = find_images(source, args.max_images, args.stride)
    image_manifest = prepare_images(image_paths, source, out)
    (out / "images_manifest.json").write_text(json.dumps(image_manifest, indent=2) + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("VGGT backend requires CUDA for this goal-sized reconstruction.")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    copied_paths = [str(out / item["copied"]) for item in image_manifest]
    images = load_and_preprocess_images(copied_paths).to(device)

    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    model.eval()
    model.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    depth_map = to_numpy(predictions["depth"])
    depth_conf = to_numpy(predictions["depth_conf"])
    extrinsic_np = to_numpy(extrinsic)
    intrinsic_np = to_numpy(intrinsic)
    world_points = unproject_depth_map_to_point_map(depth_map, extrinsic_np, intrinsic_np)
    colors_chw = images.detach().cpu().float().numpy()

    points, colors, filter_summary = filter_points(
        world_points=world_points,
        depth_conf=depth_conf,
        colors_chw=colors_chw,
        conf_percentile=args.conf_percentile,
        trim_percentile=args.trim_percentile,
        max_points=args.max_points,
        seed=args.seed,
    )
    if len(points) == 0:
        raise RuntimeError("VGGT produced no exportable points after filtering.")

    cloud = trimesh.PointCloud(points, colors=colors)
    cloud.export(out / "points.ply")
    cloud.export(out / "scene.glb")
    write_preview(points, colors, out / "preview_point_cloud.png", args.seed)

    elapsed = time.time() - start
    summary = {
        "schema_version": "0.1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "ok",
        "backend": "vggt_depth_camera_feedforward",
        "lidar_used": False,
        "legacy_lidar_assets_used": False,
        "input_source": str(source),
        "input_image_count": len(image_manifest),
        "device": device,
        "dtype": str(dtype),
        "elapsed_seconds": elapsed,
        "outputs": {
            "points_ply": "points.ply",
            "scene_glb": "scene.glb",
            "preview": "preview_point_cloud.png",
            "quality_report": "quality_report.md",
            "images_manifest": "images_manifest.json",
        },
        **filter_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(out, summary, image_manifest, elapsed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
