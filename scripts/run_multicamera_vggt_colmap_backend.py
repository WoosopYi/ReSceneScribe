#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_AGENTS = [
    "ego_vehicle",
    "ego_vehicle_behind",
    "other_vehicle",
    "other_vehicle_behind",
]
DEFAULT_CAMERAS = [
    "Camera_Front",
    "Camera_FrontLeft",
    "Camera_FrontRight",
    "Camera_Back",
    "Camera_BackLeft",
    "Camera_BackRight",
]


@dataclass(frozen=True)
class ViewRecord:
    view_id: str
    agent: str
    camera: str
    frame: int
    rgb_path: str
    mask_path: str
    c2w: np.ndarray
    intrinsic_raw: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and optionally run a VGGT -> COLMAP(+BA) backend for "
            "DeepAccident multi-agent, multi-camera reconstruction."
        )
    )
    parser.add_argument(
        "--calibrated-output",
        type=Path,
        default=Path("outputs/town04_type1_subtype2_multicam_export"),
        help="Output from run_multicam_world_reconstruction.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/town04_type1_subtype2_vggt_colmap_ba"),
    )
    parser.add_argument("--vggt-root", type=Path, default=Path("third_party/vggt"))
    parser.add_argument("--python", type=Path, default=Path(".venv-camera-only/bin/python"))
    parser.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS)
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=56)
    parser.add_argument(
        "--per-stream-limit",
        type=int,
        default=2,
        help="Evenly sample this many frames for each agent/camera stream.",
    )
    parser.add_argument("--max-images", type=int, default=48)
    parser.add_argument(
        "--image-source",
        choices=["masked", "rgb"],
        default="masked",
        help="Use dynamic-object masked images for geometry, or raw RGB.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the COLMAP scene folder and commands; do not run VGGT.",
    )
    parser.add_argument("--no-ba", action="store_true", help="Disable VGGT track BA.")
    parser.add_argument("--max-query-pts", type=int, default=2048)
    parser.add_argument("--query-frame-num", type=int, default=5)
    parser.add_argument("--max-reproj-error", type=float, default=6.0)
    parser.add_argument("--vis-thresh", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_views(calibrated_output: Path) -> list[ViewRecord]:
    cameras_path = calibrated_output / "reconstruction" / "cameras.json"
    data = read_json(cameras_path)
    views = []
    for item in data["views"]:
        views.append(
            ViewRecord(
                view_id=str(item["view_id"]),
                agent=str(item["agent"]),
                camera=str(item["camera"]),
                frame=int(item["frame"]),
                rgb_path=str(item["rgb_path"]),
                mask_path=str(item["mask_path"]),
                c2w=np.asarray(item["camera_to_world_cv"], dtype=np.float64),
                intrinsic_raw=np.asarray(item["intrinsic_raw"], dtype=np.float64),
            )
        )
    return views


def evenly_pick(items: list[ViewRecord], limit: int) -> list[ViewRecord]:
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    idxs = np.linspace(0, len(items) - 1, limit)
    return [items[int(round(i))] for i in idxs]


def select_views(views: list[ViewRecord], args: argparse.Namespace) -> list[ViewRecord]:
    agent_set = set(args.agents)
    camera_set = set(args.cameras)
    candidates = [
        v
        for v in views
        if v.agent in agent_set
        and v.camera in camera_set
        and args.frame_start <= v.frame <= args.frame_end
    ]
    groups: dict[tuple[str, str], list[ViewRecord]] = {}
    for view in sorted(candidates, key=lambda v: (v.agent, v.camera, v.frame)):
        groups.setdefault((view.agent, view.camera), []).append(view)

    per_stream: list[list[ViewRecord]] = []
    for key in sorted(groups):
        per_stream.append(evenly_pick(groups[key], args.per_stream_limit))

    selected: list[ViewRecord] = []
    cursor = 0
    while args.max_images <= 0 or len(selected) < args.max_images:
        added = False
        for group in per_stream:
            if cursor < len(group):
                selected.append(group[cursor])
                added = True
                if args.max_images > 0 and len(selected) >= args.max_images:
                    break
        if not added:
            break
        cursor += 1

    return sorted(selected, key=lambda v: (v.frame, v.agent, v.camera))


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def safe_image_name(index: int, view: ViewRecord, suffix: str) -> str:
    return f"{index:05d}__{view.agent}__{view.camera}__frame_{view.frame:03d}{suffix.lower()}"


def prepare_scene(calibrated_output: Path, out: Path, selected: list[ViewRecord], image_source: str) -> dict:
    scene = out / "scene"
    if scene.exists():
        shutil.rmtree(scene)
    for rel in ["images", "images_fullrgb", "reports"]:
        (scene / rel).mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx, view in enumerate(selected):
        rgb_src = calibrated_output / view.rgb_path
        if image_source == "masked":
            masked = calibrated_output / "masked_frames" / view.agent / view.camera / f"frame_{view.frame:03d}.jpg"
            geom_src = masked if masked.exists() else rgb_src
        else:
            geom_src = rgb_src
        suffix = geom_src.suffix if geom_src.suffix else ".jpg"
        name = safe_image_name(idx, view, suffix)
        geom_dst = scene / "images" / name
        rgb_dst = scene / "images_fullrgb" / name
        shutil.copy2(geom_src, geom_dst)
        shutil.copy2(rgb_src, rgb_dst)
        width, height = image_size(rgb_src)
        manifest.append(
            {
                "index": idx,
                "image_name": name,
                "view_id": view.view_id,
                "agent": view.agent,
                "camera": view.camera,
                "frame": view.frame,
                "geometry_image": str(geom_src.relative_to(calibrated_output)),
                "fullrgb_image": str(rgb_src.relative_to(calibrated_output)),
                "width": width,
                "height": height,
                "camera_center_world": [float(x) for x in view.c2w[:3, 3]],
                "camera_to_world_cv": view.c2w.tolist(),
                "intrinsic_raw": view.intrinsic_raw.tolist(),
            }
        )

    write_json(scene / "source_views.json", {"views": manifest})
    return {
        "scene_dir": str(scene),
        "image_count": len(manifest),
        "image_source": image_source,
        "images": "scene/images",
        "images_fullrgb": "scene/images_fullrgb",
        "source_views": "scene/source_views.json",
    }


def run_vggt_colmap(args: argparse.Namespace, scene_dir: Path, out: Path) -> dict:
    script = args.vggt_root / "demo_colmap.py"
    cmd = [
        str(args.python),
        str(script),
        f"--scene_dir={scene_dir}",
        f"--seed={args.seed}",
        f"--max_reproj_error={args.max_reproj_error}",
        f"--vis_thresh={args.vis_thresh}",
        f"--query_frame_num={args.query_frame_num}",
        f"--max_query_pts={args.max_query_pts}",
    ]
    if not args.no_ba:
        cmd.append("--use_ba")

    log_path = out / "logs" / "vggt_colmap.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.run(cmd, cwd=str(args.vggt_root), stdout=log_file, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(f"VGGT COLMAP failed with code {proc.returncode}; see {log_path}")

    sparse = scene_dir / "sparse"
    sparse0 = scene_dir / "sparse" / "0"
    if not sparse0.exists():
        sparse0.mkdir(parents=True, exist_ok=True)
        for name in ["cameras.bin", "images.bin", "points3D.bin", "points.ply"]:
            src = sparse / name
            if src.exists():
                shutil.copy2(src, sparse0 / name)

    return {
        "command": cmd,
        "bundle_adjustment": not args.no_ba,
        "log": str(log_path.relative_to(out)),
        "elapsed_sec": elapsed,
        "sparse": "scene/sparse",
        "sparse0": "scene/sparse/0",
    }


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict]:
    if src.shape[0] < 3:
        raise ValueError("Need at least three paired cameras for Sim(3) alignment.")
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mu = src.mean(axis=0)
    dst_mu = dst.mean(axis=0)
    src_c = src - src_mu
    dst_c = dst - dst_mu
    cov = (dst_c.T @ src_c) / src.shape[0]
    u, svals, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1.0
    rot = u @ d @ vt
    src_var = float(np.mean(np.sum(src_c * src_c, axis=1)))
    scale = float(np.trace(np.diag(svals) @ d) / max(src_var, 1e-12))
    trans = dst_mu - scale * (rot @ src_mu)
    aligned = (scale * (rot @ src.T)).T + trans
    residual = np.linalg.norm(aligned - dst, axis=1)
    report = {
        "pair_count": int(src.shape[0]),
        "rmse_m": float(np.sqrt(np.mean(residual * residual))),
        "median_error_m": float(np.median(residual)),
        "max_error_m": float(np.max(residual)),
    }
    return scale, rot, trans, report


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            r, g, b = [int(np.clip(x, 0, 255)) for x in color]
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n")


def alignment_report(scene_dir: Path, out: Path) -> dict:
    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("pycolmap is required for alignment reporting.") from exc

    manifest = read_json(scene_dir / "source_views.json")["views"]
    by_name = {item["image_name"]: item for item in manifest}
    sparse = scene_dir / "sparse"
    recon = pycolmap.Reconstruction(str(sparse))

    src_centers = []
    dst_centers = []
    pairs = []
    for image_id, image in recon.images.items():
        if hasattr(image, "has_pose") and not image.has_pose:
            continue
        if hasattr(image, "registered") and not image.registered:
            continue
        source = by_name.get(image.name)
        if source is None:
            continue
        src = np.asarray(image.projection_center(), dtype=np.float64)
        dst = np.asarray(source["camera_center_world"], dtype=np.float64)
        src_centers.append(src)
        dst_centers.append(dst)
        pairs.append(
            {
                "image_id": int(image_id),
                "image_name": image.name,
                "view_id": source["view_id"],
                "agent": source["agent"],
                "camera": source["camera"],
                "frame": int(source["frame"]),
            }
        )

    src_arr = np.asarray(src_centers, dtype=np.float64)
    dst_arr = np.asarray(dst_centers, dtype=np.float64)
    scale, rot, trans, residual_report = fit_similarity(src_arr, dst_arr)

    point_xyz = []
    point_rgb = []
    for _point_id, point in recon.points3D.items():
        xyz = np.asarray(point.xyz, dtype=np.float64)
        if not np.isfinite(xyz).all():
            continue
        aligned = scale * (rot @ xyz) + trans
        point_xyz.append(aligned)
        point_rgb.append(np.asarray(point.color, dtype=np.uint8))
    aligned_ply = out / "alignment" / "points3D_aligned_world.ply"
    if point_xyz:
        write_ascii_ply(aligned_ply, np.asarray(point_xyz), np.asarray(point_rgb))

    report = {
        "method": "VGGT_COLMAP_aligned_to_DeepAccident_camera_centers",
        "pair_count": len(pairs),
        "colmap_registered_images": int(len(recon.images)),
        "colmap_points3D": int(len(recon.points3D)),
        "sim3_colmap_to_deepaccident": {
            "scale": scale,
            "rotation": rot.tolist(),
            "translation": trans.tolist(),
        },
        "residuals": residual_report,
        "paired_images": pairs,
        "aligned_points_ply": str(aligned_ply.relative_to(out)) if point_xyz else None,
    }
    write_json(out / "alignment" / "sim3_alignment_report.json", report)
    return report


def write_method_notes(out: Path, summary: dict) -> None:
    notes = f"""# Multi-camera VGGT/COLMAP Backend

This backend is added because direct 3DGS training from the calibrated
DeepAccident transforms was producing blurred, low-density, poorly aligned
results. The applied change follows current multi-camera reconstruction
practice more closely:

1. Use synchronized multi-agent, six-camera RGB images in one scene folder.
2. Use dynamic-object masked images for geometric matching/reconstruction.
3. Export a COLMAP sparse model through VGGT, with optional track-based bundle
   adjustment.
4. Align the resulting COLMAP camera centers back to DeepAccident world
   coordinates using a Sim(3) transform.
5. Use the COLMAP sparse model as input to 3DGS/3DGUT instead of random or
   weakly aligned point initialization.

No `lidar01` point geometry is used. Calibration is used only as pose/scale
metadata and as the final world-frame alignment target.

## Current run

- Images: {summary.get("image_count")}
- Geometry image source: {summary.get("image_source")}
- Scene directory: `{summary.get("scene_dir")}`
- COLMAP sparse model: `{summary.get("vggt", {}).get("sparse")}`
- Alignment report: `alignment/sim3_alignment_report.json`

## Next training commands

Nerfstudio Splatfacto from COLMAP:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-camera-only/bin/ns-train splatfacto \\
  --output-dir outputs/nerfstudio_town04_vggt_colmap_ba \\
  --experiment-name town04_type1_subtype2_vggt_colmap_ba \\
  --timestamp fullrgb_colmap_ba \\
  --vis tensorboard \\
  --max-num-iterations 30000 \\
  --pipeline.datamanager.camera-res-scale-factor 0.5 \\
  --pipeline.datamanager.images-on-gpu True \\
  colmap \\
  --data {summary.get("scene_dir")} \\
  --images-path images_fullrgb \\
  --colmap-path sparse
```

3DGRUT/3DGUT from the same COLMAP folder, after installing `nv-tlabs/3dgrut`:

```bash
python train.py --config-name apps/colmap_3dgut_mcmc.yaml \\
  path={summary.get("scene_dir")} \\
  out_dir=outputs/3dgrut_town04_vggt_colmap_ba \\
  experiment_name=town04_type1_subtype2_3dgut \\
  export_usdz.enabled=false
```
"""
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "reports" / "method_notes.md").write_text(notes, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.vggt_root.is_absolute():
        args.vggt_root = (Path.cwd() / args.vggt_root).resolve()
    if not args.python.is_absolute():
        # Keep the venv entrypoint path instead of resolving the symlink to the
        # system interpreter; otherwise subprocesses lose the venv site-packages.
        args.python = (Path.cwd() / args.python).absolute()
    calibrated_output = args.calibrated_output.resolve()
    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    for rel in ["logs", "reports", "alignment"]:
        (out / rel).mkdir(parents=True, exist_ok=True)

    all_views = load_views(calibrated_output)
    selected = select_views(all_views, args)
    if len(selected) < 3:
        raise RuntimeError("Need at least 3 images for COLMAP/VGGT alignment.")

    scene_summary = prepare_scene(calibrated_output, out, selected, args.image_source)
    summary = {
        "created_at_unix": time.time(),
        "backend": "vggt_colmap_ba" if not args.no_ba else "vggt_colmap_feedforward",
        "lidar_used": False,
        "legacy_lidar_assets_used": False,
        "calibrated_output": str(calibrated_output),
        **scene_summary,
        "selection": {
            "agents": args.agents,
            "cameras": args.cameras,
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "per_stream_limit": args.per_stream_limit,
            "max_images": args.max_images,
        },
    }

    if args.prepare_only:
        summary["status"] = "prepared_only"
    else:
        vggt_summary = run_vggt_colmap(args, out / "scene", out)
        summary["vggt"] = vggt_summary
        summary["alignment"] = alignment_report(out / "scene", out)
        summary["status"] = "complete"

    write_json(out / "manifest.json", summary)
    write_method_notes(out, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
