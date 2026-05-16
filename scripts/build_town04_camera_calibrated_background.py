#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import run_camera_only_reconstruction as base
import run_multicam_world_reconstruction as multi


SCENARIO = "Town04_type001_subtype0002_scenario00017"
CATEGORY = "type1_subtype2_accident"
AGENTS = [
    "ego_vehicle",
    "ego_vehicle_behind",
    "other_vehicle",
    "other_vehicle_behind",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a no-LiDAR Town04 visual background using RGB images and calibration poses only."
    )
    parser.add_argument("--calibrated-output", type=Path, default=Path("outputs/town04_type1_subtype2_multicam_export"))
    parser.add_argument("--dataset", type=Path, default=Path("deepaccident_mini_dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs/town04_type1_subtype2_camera_background"))
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--agents", nargs="+", default=AGENTS)
    parser.add_argument("--pixel-stride", type=int, default=10)
    parser.add_argument("--max-ground-distance-m", type=float, default=95.0)
    parser.add_argument("--shell-distance-m", type=float, default=55.0)
    parser.add_argument("--min-shell-z", type=float, default=0.2)
    parser.add_argument("--max-shell-z", type=float, default=35.0)
    parser.add_argument("--max-points", type=int, default=1_400_000)
    parser.add_argument("--voxel", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--nominal-fps", type=float, default=20.0)
    return parser.parse_args()


def reset_output(out: Path) -> None:
    if out.exists():
        for rel in ["reconstruction", "reports", "replay", "viewer"]:
            target = out / rel
            if target.exists():
                shutil.rmtree(target)
        for rel in ["scene.glb", "summary.json", "manifest.json"]:
            target = out / rel
            if target.exists():
                target.unlink()
    for rel in ["reconstruction", "reports", "replay", "viewer"]:
        (out / rel).mkdir(parents=True, exist_ok=True)


def standard_intrinsics(raw: np.ndarray) -> tuple[float, float, float, float]:
    raw = np.asarray(raw, dtype=np.float64)
    cx = float(raw[0, 0])
    cy = float(raw[1, 0])
    fx = abs(float(raw[0, 1]))
    fy = abs(float(raw[1, 2]))
    if fx <= 1:
        fx = fy
    if fy <= 1:
        fy = fx
    return fx, fy, cx, cy


def pixel_grid(width: int, height: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(stride // 2, width, stride, dtype=np.float64)
    ys = np.arange(stride // 2, height, stride, dtype=np.float64)
    return np.meshgrid(xs, ys)


def is_sky_like(colors: np.ndarray, ys: np.ndarray, height: int) -> np.ndarray:
    rgb = colors.astype(np.int16)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    high = ys < height * 0.58
    blue_sky = (b > 115) & (b > r + 18) & (b > g + 8)
    pale_sky = (b > 145) & (g > 135) & (r > 115) & (np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b]) < 45)
    return high & (blue_sky | pale_sky)


def sample_view(view: dict, calibrated_output: Path, stride: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    masked_rel = Path("masked_frames") / view["agent"] / view["camera"] / f"frame_{int(view['frame']):03d}.jpg"
    masked_path = calibrated_output / masked_rel
    if not masked_path.exists():
        rgb_rel = Path(view["rgb_path"])
        masked_path = calibrated_output / rgb_rel
    img = np.asarray(Image.open(masked_path).convert("RGB"))
    height, width = img.shape[:2]
    xx, yy = pixel_grid(width, height, stride)
    ui = xx.reshape(-1)
    vi = yy.reshape(-1)
    px = np.clip(np.rint(ui).astype(np.int32), 0, width - 1)
    py = np.clip(np.rint(vi).astype(np.int32), 0, height - 1)
    colors = img[py, px]

    nonblack = colors.sum(axis=1) > 45
    nonsky = ~is_sky_like(colors, vi, height)
    keep_color = nonblack & nonsky
    if not np.any(keep_color):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), {
            "view_id": view["view_id"],
            "ground_points": 0,
            "shell_points": 0,
        }

    ui = ui[keep_color]
    vi = vi[keep_color]
    colors = colors[keep_color]

    fx, fy, cx, cy = standard_intrinsics(np.asarray(view["intrinsic_raw"], dtype=np.float64))
    dirs_cv = np.column_stack([(ui - cx) / fx, (vi - cy) / fy, np.ones_like(ui)])
    dirs_cv /= np.maximum(np.linalg.norm(dirs_cv, axis=1, keepdims=True), 1e-9)
    c2w = np.asarray(view["camera_to_world_cv"], dtype=np.float64)
    origin = c2w[:3, 3]
    dirs_world = (c2w[:3, :3] @ dirs_cv.T).T
    dirs_world /= np.maximum(np.linalg.norm(dirs_world, axis=1, keepdims=True), 1e-9)

    points = []
    point_colors = []

    downward = dirs_world[:, 2] < -0.015
    t_ground = (0.02 - origin[2]) / np.minimum(dirs_world[:, 2], -1e-9)
    ground = downward & (t_ground > 0.5) & (t_ground < args.max_ground_distance_m)
    if np.any(ground):
        p = origin[None, :] + dirs_world[ground] * t_ground[ground, None]
        points.append(p.astype(np.float32))
        point_colors.append(colors[ground])

    shell_src = ~ground
    horizontal_norm = np.linalg.norm(dirs_world[:, :2], axis=1)
    t_shell = args.shell_distance_m / np.maximum(horizontal_norm, 1e-6)
    shell_points = origin[None, :] + dirs_world * t_shell[:, None]
    shell = (
        shell_src
        & (t_shell > 4.0)
        & np.isfinite(shell_points).all(axis=1)
        & (shell_points[:, 2] >= args.min_shell_z)
        & (shell_points[:, 2] <= args.max_shell_z)
    )
    if np.any(shell):
        p = shell_points[shell]
        points.append(p.astype(np.float32))
        point_colors.append(colors[shell])

    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), {
            "view_id": view["view_id"],
            "ground_points": 0,
            "shell_points": 0,
        }

    return np.vstack(points), np.vstack(point_colors), {
        "view_id": view["view_id"],
        "ground_points": int(np.count_nonzero(ground)),
        "shell_points": int(np.count_nonzero(shell)),
    }


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel: float, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, colors
    if voxel > 0:
        keys = np.floor(points / voxel).astype(np.int64)
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        points = points[unique_idx]
        colors = colors[unique_idx]
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points = points[idx]
        colors = colors[idx]
    return points, colors


def write_topdown_preview(path: Path, points: np.ndarray, colors: np.ndarray, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = np.arange(len(points))
    if len(idx) > 250_000:
        idx = np.sort(rng.choice(idx, size=250_000, replace=False))
    pts = points[idx]
    rgb = colors[idx] / 255.0
    fig, ax = plt.subplots(figsize=(12, 10), dpi=160)
    ax.scatter(pts[:, 0], pts[:, 1], c=rgb, s=0.08, linewidths=0, alpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_title("Town04 no-LiDAR calibrated RGB background top-down")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(out: Path, args: argparse.Namespace, summary: dict) -> None:
    report = f"""# Town04 Camera-Calibrated Background

Status: `{summary['status']}`

## Method

- Geometry source: RGB camera rays plus DeepAccident calibration only.
- `lidar01`: not used.
- Dynamic suppression: uses pre-existing masked RGB frames from `{args.calibrated_output}`.
- Ground: image rays intersected with a fixed world ground plane.
- Distant background: non-sky image rays projected onto a bounded shell at `{args.shell_distance_m}` m.

This is a stable visual background proxy, not a metric dense MVS reconstruction.

## Inputs

- Scenario: `{args.scenario}`
- Input views: `{summary['input_view_count']}`
- Pixel stride: `{args.pixel_stride}`

## Outputs

- Exported points: `{summary['exported_point_count']}`
- Ground candidate points: `{summary['ground_candidate_points']}`
- Shell candidate points: `{summary['shell_candidate_points']}`
- World bbox min: `{summary['bbox_min_world']}`
- World bbox max: `{summary['bbox_max_world']}`
- `scene.glb`
- `reconstruction/points_world.ply`
- `reports/topdown_preview.png`
- `viewer/index.html`
"""
    (out / "reports" / "quality_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    base.CATEGORY = args.category
    args.dataset = args.dataset.resolve()
    args.calibrated_output = args.calibrated_output.resolve()
    args.out = args.out.resolve()
    reset_output(args.out)
    start = time.time()

    cameras_path = args.calibrated_output / "reconstruction" / "cameras.json"
    camera_data = json.loads(cameras_path.read_text(encoding="utf-8"))
    views = camera_data["views"]
    point_parts = []
    color_parts = []
    view_reports = []
    ground_total = 0
    shell_total = 0

    for view in views:
        points, colors, report = sample_view(view, args.calibrated_output, args.pixel_stride, args)
        if len(points):
            point_parts.append(points)
            color_parts.append(colors)
        view_reports.append(report)
        ground_total += report["ground_points"]
        shell_total += report["shell_points"]

    if not point_parts:
        raise RuntimeError("No RGB-calibrated background points generated.")

    points_world = np.vstack(point_parts)
    colors = np.vstack(color_parts)
    before_voxel = int(len(points_world))
    points_world, colors = voxel_downsample(points_world, colors, args.voxel, args.max_points, args.seed)
    viewer_points, origin = base.world_to_viewer(points_world)

    base.write_ply(args.out / "reconstruction" / "points_world.ply", points_world, colors)
    base.write_ply(args.out / "reconstruction" / "points.ply", viewer_points, colors)
    base.export_scene_glb(args.out / "scene.glb", viewer_points, colors)
    base.write_preview(args.out / "reports" / "preview_point_cloud.png", viewer_points, colors)
    write_topdown_preview(args.out / "reports" / "topdown_preview.png", points_world, colors, args.seed)
    base.write_json(args.out / "reconstruction" / "view_reports.json", {"views": view_reports})
    shutil.copy2(cameras_path, args.out / "reconstruction" / "cameras.json")

    frames = sorted({int(v["frame"]) for v in views})
    tracks = multi.build_agent_tracks(args.dataset, args.out, args.agents, frames, args.scenario, origin, args.nominal_fps)
    object_context = multi.build_object_tracks(args.dataset, args.out, args.agents, frames, args.scenario)
    multi.accident_diagnostics(args.out, tracks)
    multi.write_viewer(args.out)

    summary = {
        "schema_version": "0.1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "ok",
        "backend": "camera_calibrated_ground_and_background_proxy",
        "lidar_used": False,
        "calibration_used": True,
        "dataset_root": str(args.dataset),
        "calibrated_output": str(args.calibrated_output),
        "scenario": args.scenario,
        "input_view_count": len(views),
        "ground_candidate_points": int(ground_total),
        "shell_candidate_points": int(shell_total),
        "points_before_voxel": before_voxel,
        "exported_point_count": int(len(points_world)),
        "world_origin": origin.tolist(),
        "bbox_min_world": [float(x) for x in points_world.min(axis=0)],
        "bbox_max_world": [float(x) for x in points_world.max(axis=0)],
        "object_context_records": int(object_context["record_count"]),
        "elapsed_seconds": time.time() - start,
        "outputs": {
            "points_world_ply": "reconstruction/points_world.ply",
            "points_ply": "reconstruction/points.ply",
            "scene_glb": "scene.glb",
            "preview": "reports/preview_point_cloud.png",
            "topdown_preview": "reports/topdown_preview.png",
            "quality_report": "reports/quality_report.md",
            "viewer": "viewer/index.html",
            "agent_tracks": "replay/agent_tracks.json",
            "accident_diagnostics": "replay/accident_diagnostics.json",
        },
    }
    base.write_json(args.out / "summary.json", summary)
    manifest = {
        "schema_version": "0.1",
        "case_id": "town04_type1_subtype2_camera_background",
        "status": "ok",
        "created_utc": summary["created_utc"],
        "inputs": {
            "calibrated_output": str(args.calibrated_output),
            "dataset_root": str(args.dataset),
            "scenario": args.scenario,
            "input_view_count": len(views),
        },
        "outputs": summary["outputs"],
        "quality_summary": {
            "backend": summary["backend"],
            "point_count": summary["exported_point_count"],
            "points_before_voxel": summary["points_before_voxel"],
            "ground_candidate_points": summary["ground_candidate_points"],
            "shell_candidate_points": summary["shell_candidate_points"],
        },
        "lidar_used": False,
        "rgb_used": True,
        "calibration_used": True,
    }
    base.write_json(args.out / "manifest.json", manifest)
    write_report(args.out, args, summary)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
