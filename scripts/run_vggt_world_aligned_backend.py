#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

import run_camera_only_reconstruction as base
import run_multicam_world_reconstruction as multi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT dense reconstruction and align it to DeepAccident calibration world coordinates."
    )
    parser.add_argument("--calibrated-output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("deepaccident_mini_dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs/multicam_world_reconstruction_vggt"))
    parser.add_argument("--scenario", default=base.SCENARIO)
    parser.add_argument("--category", default=base.DEFAULT_CATEGORY)
    parser.add_argument("--agents", nargs="+", default=multi.VEHICLE_AGENTS)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=56)
    parser.add_argument("--frame-step", type=int, default=7)
    parser.add_argument("--vggt-root", type=Path, default=Path("third_party/vggt"))
    parser.add_argument("--max-images", type=int, default=216)
    parser.add_argument("--group-mode", choices=["global", "agent"], default="global")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--conf-percentile", type=float, default=55.0)
    parser.add_argument("--trim-percentile", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=800_000)
    parser.add_argument("--nominal-fps", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def parse_masked_image_path(path: Path) -> tuple[str, str, int] | None:
    parts = path.parts
    if len(parts) < 3:
        return None
    agent = parts[-3]
    camera = parts[-2]
    stem = path.stem
    if not stem.startswith("frame_"):
        return None
    try:
        frame = int(stem.replace("frame_", ""))
    except ValueError:
        return None
    return agent, camera, frame


def select_balanced_images(source: Path, max_images: int, stride: int) -> list[Path]:
    candidates = sorted(
        p
        for p in source.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()
    )
    if stride > 1:
        candidates = candidates[::stride]
    if max_images <= 0 or len(candidates) <= max_images:
        return candidates

    groups: dict[tuple[str, str], list[Path]] = {}
    for path in candidates:
        parsed = parse_masked_image_path(path)
        if parsed is None:
            continue
        agent, camera, _frame = parsed
        groups.setdefault((agent, camera), []).append(path)

    selected = []
    group_items = sorted(groups.items())
    cursor = 0
    while len(selected) < max_images:
        added = False
        for _key, paths in group_items:
            if cursor < len(paths):
                selected.append(paths[cursor])
                added = True
                if len(selected) >= max_images:
                    break
        if not added:
            break
        cursor += 1
    return sorted(selected)


def prepare_images(paths: list[Path], source: Path, out: Path) -> list[dict]:
    image_dir = out / "vggt_images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, path in enumerate(paths):
        rel = path.relative_to(source)
        parsed = parse_masked_image_path(rel)
        if parsed is None:
            continue
        agent, camera, frame = parsed
        safe_name = f"{idx:05d}__{agent}__{camera}__frame_{frame:03d}{path.suffix.lower()}"
        dst = image_dir / safe_name
        shutil.copy2(path, dst)
        manifest.append(
            {
                "index": idx,
                "agent": agent,
                "camera": camera,
                "frame": frame,
                "source": str(path),
                "source_relative": str(rel),
                "copied": str(dst.relative_to(out)),
            }
        )
    if not manifest:
        raise FileNotFoundError(f"No usable masked RGB images found under {source}")
    base.write_json(out / "vggt_images_manifest.json", {"images": manifest})
    return manifest


def load_calibrated_camera_centers(calibrated_output: Path) -> dict[tuple[str, str, int], np.ndarray]:
    cameras_path = calibrated_output / "reconstruction" / "cameras.json"
    data = json.loads(cameras_path.read_text(encoding="utf-8"))
    centers = {}
    for view in data["views"]:
        c2w = np.asarray(view["camera_to_world_cv"], dtype=np.float64)
        centers[(view["agent"], view["camera"], int(view["frame"]))] = c2w[:3, 3]
    return centers


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().float().numpy().squeeze(0)


def camera_centers_from_world_to_cam(extrinsics: np.ndarray) -> np.ndarray:
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    return np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), -translations)


def fit_similarity_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 3:
        raise ValueError("Need at least three camera centers for similarity alignment.")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / len(src)
    u, svals, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1.0
    rot = u @ d @ vt
    var_src = float(np.mean(np.sum(src_c * src_c, axis=1)))
    scale = float(np.trace(np.diag(svals) @ d) / max(var_src, 1e-12))
    trans = mu_dst - scale * (rot @ mu_src)
    aligned = (scale * (rot @ src.T)).T + trans
    residual = np.linalg.norm(aligned - dst, axis=1)
    report = {
        "camera_count": int(len(src)),
        "rmse_m": float(np.sqrt(np.mean(residual * residual))),
        "median_error_m": float(np.median(residual)),
        "max_error_m": float(np.max(residual)),
        "errors_m": [float(x) for x in residual],
    }
    return scale, rot, trans, report


def apply_similarity(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3).astype(np.float64)
    aligned = (scale * (rot @ flat.T)).T + trans
    return aligned.reshape(points.shape).astype(np.float32)


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
    if len(selected_points):
        summary["bbox_min_world"] = [float(x) for x in selected_points.min(axis=0)]
        summary["bbox_max_world"] = [float(x) for x in selected_points.max(axis=0)]
    return selected_points, selected_colors, summary


def write_preview(path: Path, points: np.ndarray, colors: np.ndarray, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(points) == 0:
        return
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), min(80_000, len(points)), replace=False)
    pts = points[idx]
    rgb = colors[idx] / 255.0
    fig = plt.figure(figsize=(11, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=rgb, s=0.25, linewidths=0, alpha=0.9)
    center = pts.mean(axis=0)
    radius = max(float(np.percentile(np.linalg.norm(pts - center, axis=1), 95)), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)
    ax.set_xlabel("viewer x")
    ax.set_ylabel("viewer z")
    ax.set_zlabel("viewer y/up")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def copy_calibrated_artifacts(calibrated_output: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    for rel in ["rgb_frames", "masks", "masked_frames", "nerfstudio"]:
        src = calibrated_output / rel
        if src.exists():
            shutil.copytree(src, out / rel)
    for rel in ["mask_manifest.json"]:
        src = calibrated_output / rel
        if src.exists():
            shutil.copy2(src, out / rel)
    for rel in ["reconstruction", "reports", "replay", "viewer"]:
        (out / rel).mkdir(parents=True, exist_ok=True)
    src_cameras = calibrated_output / "reconstruction" / "cameras.json"
    if src_cameras.exists():
        shutil.copy2(src_cameras, out / "reconstruction" / "cameras.json")


def aggregate_alignment_residual(group_alignments: list[dict]) -> dict:
    errors = []
    for alignment in group_alignments:
        errors.extend(alignment["residual"].get("errors_m", []))
    if not errors:
        return {"camera_count": 0, "rmse_m": None, "median_error_m": None, "max_error_m": None}
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "camera_count": int(len(arr)),
        "rmse_m": float(np.sqrt(np.mean(arr * arr))),
        "median_error_m": float(np.median(arr)),
        "max_error_m": float(np.max(arr)),
    }


def run_vggt_group(
    *,
    group_name: str,
    group_manifest: list[dict],
    out: Path,
    model: torch.nn.Module,
    load_and_preprocess_images,
    pose_encoding_to_extri_intri,
    unproject_depth_map_to_point_map,
    target_centers: np.ndarray,
    dtype: torch.dtype,
    args: argparse.Namespace,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    copied_paths = [str(out / item["copied"]) for item in group_manifest]
    images = load_and_preprocess_images(copied_paths).to("cuda")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    depth_map = to_numpy(predictions["depth"])
    depth_conf = to_numpy(predictions["depth_conf"])
    extrinsic_np = to_numpy(extrinsic)
    intrinsic_np = to_numpy(intrinsic)
    vggt_world_points = unproject_depth_map_to_point_map(depth_map, extrinsic_np, intrinsic_np)
    predicted_centers = camera_centers_from_world_to_cam(extrinsic_np)
    scale, rot, trans, residual = fit_similarity_umeyama(predicted_centers, target_centers)
    aligned_world_points = apply_similarity(vggt_world_points, scale, rot, trans)
    colors_chw = images.detach().cpu().float().numpy()
    points_world, colors, filter_summary = filter_points(
        world_points=aligned_world_points,
        depth_conf=depth_conf,
        colors_chw=colors_chw,
        conf_percentile=args.conf_percentile,
        trim_percentile=args.trim_percentile,
        max_points=max_points,
        seed=seed,
    )
    alignment = {
        "group": group_name,
        "method": "umeyama_similarity_fit_vggt_camera_centers_to_calibration_camera_centers",
        "scale": scale,
        "rotation": rot.tolist(),
        "translation": trans.tolist(),
        "residual": residual,
    }
    group_summary = {
        "group": group_name,
        "input_image_count": len(group_manifest),
        **filter_summary,
    }
    del images, predictions, extrinsic, intrinsic
    torch.cuda.empty_cache()
    return points_world, colors, alignment, group_summary


def write_reports(out: Path, args: argparse.Namespace, summary: dict, alignment: dict, diagnostics: dict) -> None:
    no_lidar = f"""# No-LiDAR Audit

Status: pass for `scripts/run_vggt_world_aligned_backend.py`.

Inputs used:

- Masked RGB images from `{args.calibrated_output / 'masked_frames'}`
- DeepAccident calibration camera centers from `{args.calibrated_output / 'reconstruction/cameras.json'}`
- DeepAccident `ego_to_world` calibration for vehicle trajectories
- DeepAccident label text for object context only

Inputs not used:

- `lidar01` point clouds: not read
- `viewer_assets/*.glb`: not read
- Historical LiDAR-derived PLY/GLB: not read

Alignment:

- VGGT dense scene is aligned to calibration world coordinates using camera-center Sim(3) fitting.
- Alignment RMSE: `{alignment['residual']['rmse_m']}` m
- Alignment median error: `{alignment['residual']['median_error_m']}` m
"""
    (out / "reports" / "no_lidar_audit.md").write_text(no_lidar, encoding="utf-8")

    quality = f"""# VGGT World-aligned Quality Report

Status: `{summary['status']}`

## Dense Backend

- Backend: `vggt_dense_world_aligned`
- Input images: `{summary['input_image_count']}`
- Exported points: `{summary['exported_point_count']}`
- Raw model points: `{summary['raw_point_count']}`
- Confidence percentile: `{summary['confidence_percentile']}`
- Alignment camera centers: `{alignment['residual']['camera_count']}`
- Alignment RMSE m: `{alignment['residual']['rmse_m']}`
- Alignment median error m: `{alignment['residual']['median_error_m']}`
- LiDAR geometry used: `false`

## Motion And Accident Context

- Track source: calibration `ego_to_world` per agent/frame.
- Closest proxy approach: `{diagnostics.get('closest_approach')}`
- Proxy caveat: center-distance proxy, not full physics contact.

## Outputs

- `scene.glb`
- `reconstruction/points.ply`
- `reconstruction/points_world.ply`
- `reconstruction/alignment.json`
- `replay/agent_tracks.json`
- `replay/accident_diagnostics.json`
- `viewer/index.html`
- `nerfstudio/transforms.json`
"""
    (out / "reports" / "quality_report.md").write_text(quality, encoding="utf-8")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        raise RuntimeError("VGGT world-aligned backend requires CUDA.")

    calibrated_output = args.calibrated_output.resolve()
    base.CATEGORY = args.category
    dataset = args.dataset.resolve()
    out = args.out.resolve()
    copy_calibrated_artifacts(calibrated_output, out)

    vggt_root = args.vggt_root.resolve()
    if not vggt_root.exists():
        raise FileNotFoundError(f"VGGT root does not exist: {vggt_root}")
    sys.path.insert(0, str(vggt_root))

    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    start = time.time()
    source = calibrated_output / "masked_frames"
    image_paths = select_balanced_images(source, args.max_images, args.stride)
    image_manifest = prepare_images(image_paths, source, out)
    calibration_centers = load_calibrated_camera_centers(calibrated_output)
    target_centers = []
    for item in image_manifest:
        key = (item["agent"], item["camera"], int(item["frame"]))
        target_centers.append(calibration_centers[key])
    target_centers = np.asarray(target_centers, dtype=np.float64)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, progress=True))
    model.eval()
    model.to("cuda")

    if args.group_mode == "agent":
        grouped: dict[str, list[dict]] = {}
        for item in image_manifest:
            grouped.setdefault(item["agent"], []).append(item)
        group_items = sorted(grouped.items())
    else:
        group_items = [("global", image_manifest)]

    points_parts = []
    color_parts = []
    group_alignments = []
    group_summaries = []
    per_group_max = args.max_points
    if args.group_mode == "agent" and args.max_points > 0:
        per_group_max = max(1, int(math.ceil(args.max_points / max(1, len(group_items)) * 1.2)))

    for group_index, (group_name, group_manifest) in enumerate(group_items):
        group_target_centers = []
        for item in group_manifest:
            key = (item["agent"], item["camera"], int(item["frame"]))
            group_target_centers.append(calibration_centers[key])
        group_target_centers = np.asarray(group_target_centers, dtype=np.float64)
        group_points, group_colors, group_alignment, group_summary = run_vggt_group(
            group_name=group_name,
            group_manifest=group_manifest,
            out=out,
            model=model,
            load_and_preprocess_images=load_and_preprocess_images,
            pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
            unproject_depth_map_to_point_map=unproject_depth_map_to_point_map,
            target_centers=group_target_centers,
            dtype=dtype,
            args=args,
            max_points=per_group_max,
            seed=args.seed + group_index,
        )
        points_parts.append(group_points)
        color_parts.append(group_colors)
        group_alignments.append(group_alignment)
        group_summaries.append(group_summary)

    points_world = np.vstack(points_parts)
    colors = np.vstack(color_parts)
    if args.max_points > 0 and len(points_world) > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(points_world), size=args.max_points, replace=False))
        points_world = points_world[idx]
        colors = colors[idx]

    residual = aggregate_alignment_residual(group_alignments)
    filter_summary = {
        "raw_point_count": int(sum(s["raw_point_count"] for s in group_summaries)),
        "valid_point_count_before_sampling": int(sum(s["valid_point_count_before_sampling"] for s in group_summaries)),
        "exported_point_count": int(len(points_world)),
        "confidence_percentile": float(args.conf_percentile),
        "confidence_threshold": None,
        "trim_percentile": float(args.trim_percentile),
        "max_points": int(args.max_points),
        "group_mode": args.group_mode,
        "group_summaries": group_summaries,
    }
    if len(points_world):
        filter_summary["bbox_min_world"] = [float(x) for x in points_world.min(axis=0)]
        filter_summary["bbox_max_world"] = [float(x) for x in points_world.max(axis=0)]
    if len(points_world) == 0:
        raise RuntimeError("VGGT produced no exportable world-aligned points after filtering.")

    viewer_points, origin = base.world_to_viewer(points_world)
    base.write_ply(out / "reconstruction" / "points_world.ply", points_world, colors)
    base.write_ply(out / "reconstruction" / "points.ply", viewer_points, colors)
    trimesh.PointCloud(viewer_points.astype(np.float32), colors=colors).export(out / "scene.glb")
    write_preview(out / "reports" / "preview_point_cloud.png", viewer_points, colors, args.seed)

    frames = base.select_frames(args.frame_start, args.frame_end, args.frame_step)
    tracks = multi.build_agent_tracks(dataset, out, args.agents, frames, args.scenario, origin, args.nominal_fps)
    object_context = multi.build_object_tracks(dataset, out, args.agents, frames, args.scenario)
    diagnostics = multi.accident_diagnostics(out, tracks)
    multi.write_viewer(out)

    elapsed = time.time() - start
    alignment = {
        "method": "umeyama_similarity_fit_vggt_camera_centers_to_calibration_camera_centers",
        "group_mode": args.group_mode,
        "viewer_origin_world": origin.tolist(),
        "residual": residual,
        "groups": group_alignments,
    }
    base.write_json(out / "reconstruction" / "alignment.json", alignment)

    summary = {
        "schema_version": "0.2",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "ok",
        "backend": "vggt_dense_world_aligned",
        "lidar_used": False,
        "legacy_lidar_assets_used": False,
        "input_source": str(source),
        "input_image_count": len(image_manifest),
        "device": "cuda",
        "dtype": str(dtype),
        "elapsed_seconds": elapsed,
        "outputs": {
            "points_ply": "reconstruction/points.ply",
            "points_world_ply": "reconstruction/points_world.ply",
            "scene_glb": "scene.glb",
            "preview": "reports/preview_point_cloud.png",
            "quality_report": "reports/quality_report.md",
            "alignment": "reconstruction/alignment.json",
            "agent_tracks": "replay/agent_tracks.json",
            "accident_diagnostics": "replay/accident_diagnostics.json",
        },
        "alignment_residual": residual,
        "object_context_records": object_context["record_count"],
        **filter_summary,
    }
    base.write_json(out / "summary.json", summary)
    manifest = {
        "schema_version": "0.4",
        "case_id": "multicam_world_reconstruction",
        "status": "ok",
        "created_utc": summary["created_utc"],
        "lidar_used": False,
        "legacy_lidar_assets_used": False,
        "inputs": {
            "calibrated_output": str(calibrated_output),
            "dataset_root": str(dataset),
            "category": args.category,
            "scenario": args.scenario,
            "vggt_image_count": len(image_manifest),
        },
        "outputs": summary["outputs"],
        "quality_summary": {
            "backend": summary["backend"],
            "point_count": summary["exported_point_count"],
            "alignment_rmse_m": residual["rmse_m"],
            "alignment_median_error_m": residual["median_error_m"],
            "object_context_records": object_context["record_count"],
        },
    }
    base.write_json(out / "manifest.json", manifest)
    write_reports(out, args, summary, alignment, diagnostics)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
