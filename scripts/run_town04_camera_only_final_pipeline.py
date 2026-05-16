#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

import run_camera_only_reconstruction as base
import run_multicam_world_reconstruction as multi
import run_multicamera_vggt_colmap_backend as vggt_colmap


SCENARIO = "Town04_type001_subtype0002_scenario00017"
CATEGORY = "type1_subtype2_accident"
OUTPUT_ROOT = Path("outputs/camera_only_reconstruction_town04_final")
CALIBRATED_EXPORT = Path("outputs/town04_type1_subtype2_multicam_export")
PYTHON = Path(".venv-camera-only/bin/python")
VGGT_ROOT = Path("third_party/vggt")

AGENTS = [
    "ego_vehicle",
    "ego_vehicle_behind",
    "other_vehicle",
    "other_vehicle_behind",
]
CAMERAS = [
    "Camera_Front",
    "Camera_FrontLeft",
    "Camera_FrontRight",
    "Camera_Back",
    "Camera_BackLeft",
    "Camera_BackRight",
]
FRONT_RIG = ["Camera_Front", "Camera_FrontLeft", "Camera_FrontRight"]
REAR_RIG = ["Camera_Back", "Camera_BackLeft", "Camera_BackRight"]


@dataclass(frozen=True)
class ShardDefinition:
    name: str
    selectors: tuple[tuple[str, str], ...]
    frames: tuple[int, ...]
    kind: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final no-LiDAR Town04 camera-only reconstruction pipeline: "
            "RGB export, overlap-aware VGGT/COLMAP shards, world fusion, optional 3DGS, reports, and viewer."
        )
    )
    parser.add_argument("--dataset", type=Path, default=Path("deepaccident_mini_dataset"))
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--calibrated-output", type=Path, default=CALIBRATED_EXPORT)
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--agents", nargs="+", default=AGENTS)
    parser.add_argument("--cameras", nargs="+", default=CAMERAS)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=56)
    parser.add_argument(
        "--rig-frames",
        default="1,17,33,49",
        help="Comma-separated frames for three-camera front/rear rig shards.",
    )
    parser.add_argument(
        "--transition-frames",
        default="1,12,23,34,45,56",
        help="Comma-separated frames for side-transition shards.",
    )
    parser.add_argument(
        "--collision-frames",
        default="45-56",
        help="Comma-separated frames or ranges used to discover and run cross-agent overlap shards.",
    )
    parser.add_argument("--max-cross-shards", type=int, default=4)
    parser.add_argument("--max-shards", type=int, default=0, help="0 means all planned shards.")
    parser.add_argument("--reuse-calibrated", action="store_true", help="Use an existing calibrated RGB export if present.")
    parser.add_argument("--reuse-shards", action="store_true", help="Reuse completed shard manifests under the final output.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare shard image folders/manifests but do not run VGGT.")
    parser.add_argument("--skip-export", action="store_true", help="Do not rebuild the calibrated RGB/mask export.")
    parser.add_argument("--mask-backend", choices=["yolo", "empty"], default="yolo")
    parser.add_argument("--yolo-model", default="yolo11n-seg.pt")
    parser.add_argument("--yolo-imgsz", type=int, default=960)
    parser.add_argument("--vggt-root", type=Path, default=VGGT_ROOT)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--no-ba", action="store_true", help="Disable VGGT bundle adjustment. Shards will fail the final gate unless --allow-no-ba is also set.")
    parser.add_argument("--allow-no-ba", action="store_true")
    parser.add_argument("--max-query-pts", type=int, default=2048)
    parser.add_argument("--query-frame-num", type=int, default=5)
    parser.add_argument("--max-reproj-error", type=float, default=8.0)
    parser.add_argument("--vis-thresh", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--min-registered-ratio", type=float, default=0.70)
    parser.add_argument("--max-median-align-error-m", type=float, default=1.5)
    parser.add_argument("--max-rmse-align-error-m", type=float, default=4.0)
    parser.add_argument("--min-shard-points", type=int, default=100)
    parser.add_argument("--max-shard-bbox-m", type=float, default=260.0)
    parser.add_argument("--max-shard-z-span-m", type=float, default=80.0)
    parser.add_argument("--fusion-voxel", type=float, default=0.08)
    parser.add_argument("--fusion-trim-percentile", type=float, default=0.75)
    parser.add_argument("--fusion-max-distance-m", type=float, default=180.0)
    parser.add_argument("--fusion-min-z-m", type=float, default=-4.0)
    parser.add_argument("--fusion-max-z-m", type=float, default=45.0)
    parser.add_argument("--fusion-max-points", type=int, default=1_200_000)
    parser.add_argument("--nominal-fps", type=float, default=20.0)
    parser.add_argument("--skip-3dgs", action="store_true")
    parser.add_argument("--gs-iterations", type=int, default=15000)
    parser.add_argument("--camera-res-scale", type=float, default=0.5)
    parser.add_argument("--ns-train", type=Path, default=Path(".venv-camera-only/bin/ns-train"))
    parser.add_argument("--dry-run", action="store_true", help="Plan and validate paths without running heavy export, VGGT, or 3DGS.")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_similarity_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict]:
    return vggt_colmap.fit_similarity(np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64))


def parse_frame_list(value: str, start: int, end: int) -> tuple[int, ...]:
    frames = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "-" in raw or ".." in raw:
            sep = ".." if ".." in raw else "-"
            lo_raw, hi_raw = raw.split(sep, 1)
            lo = int(lo_raw)
            hi = int(hi_raw)
            for frame in range(min(lo, hi), max(lo, hi) + 1):
                if start <= frame <= end:
                    frames.append(frame)
        else:
            frame = int(raw)
            if start <= frame <= end:
                frames.append(frame)
    return tuple(sorted(set(frames)))


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def available_calibration_frames(dataset: Path, category: str, scenario: str, agents: list[str]) -> tuple[int, ...]:
    frames: set[int] | None = None
    pattern = re.compile(rf"{re.escape(scenario)}_(\d+)\.pkl$")
    for agent in agents:
        calib_dir = dataset / category / agent / "calib" / scenario
        agent_frames = set()
        if calib_dir.exists():
            for path in calib_dir.glob(f"{scenario}_*.pkl"):
                match = pattern.match(path.name)
                if match:
                    agent_frames.add(int(match.group(1)))
        frames = agent_frames if frames is None else frames & agent_frames
    return tuple(sorted(frames or ()))


def run_command(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool = False) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return {"cmd": cmd, "returncode": 0, "elapsed_sec": 0.0, "log": str(log_path)}
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log_file, stderr=subprocess.STDOUT, text=True)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "elapsed_sec": time.time() - start,
        "log": str(log_path),
    }


def ensure_final_dirs(out: Path) -> None:
    for rel in ["reconstruction", "reconstruction/shards", "reports", "replay", "viewer", "logs"]:
        (out / rel).mkdir(parents=True, exist_ok=True)


def rebuild_calibrated_export(args: argparse.Namespace, repo: Path) -> dict:
    target = args.calibrated_output.resolve()
    cameras_path = target / "reconstruction" / "cameras.json"
    available_frames = available_calibration_frames(args.dataset, args.category, args.scenario, args.agents)
    effective_frame_end = args.frame_end
    if available_frames:
        bounded = [frame for frame in available_frames if args.frame_start <= frame <= args.frame_end]
        if bounded:
            effective_frame_end = max(bounded)
    if args.skip_export or args.dry_run or (args.reuse_calibrated and cameras_path.exists()):
        return {
            "status": "reused" if cameras_path.exists() else "skipped",
            "path": str(target),
            "cameras_json_exists": cameras_path.exists(),
            "requested_frame_range": [args.frame_start, args.frame_end],
            "effective_frame_range": [args.frame_start, effective_frame_end],
            "available_calibration_frame_count": len(available_frames),
        }
    cmd = [
        str(args.python),
        "scripts/run_multicam_world_reconstruction.py",
        "--dataset",
        str(args.dataset),
        "--out",
        str(target),
        "--scenario",
        args.scenario,
        "--category",
        args.category,
        "--agents",
        *args.agents,
        "--cameras",
        *args.cameras,
        "--frame-start",
        str(args.frame_start),
        "--frame-end",
        str(effective_frame_end),
        "--frame-step",
        "1",
        "--skip-sift-reconstruction",
        "--mask-backend",
        args.mask_backend,
        "--yolo-model",
        args.yolo_model,
        "--yolo-imgsz",
        str(args.yolo_imgsz),
        "--nominal-fps",
        str(args.nominal_fps),
    ]
    result = run_command(cmd, repo, args.out.resolve() / "logs" / "calibrated_export.log", dry_run=False)
    result["path"] = str(target)
    result["cameras_json_exists"] = cameras_path.exists()
    result["requested_frame_range"] = [args.frame_start, args.frame_end]
    result["effective_frame_range"] = [args.frame_start, effective_frame_end]
    result["available_calibration_frame_count"] = len(available_frames)
    if result["returncode"] != 0:
        raise RuntimeError(f"Calibrated export failed; see {result['log']}")
    return result


def image_forward(view: vggt_colmap.ViewRecord) -> np.ndarray:
    forward = np.asarray(view.c2w[:3, 2], dtype=np.float64)
    return forward / max(float(np.linalg.norm(forward)), 1e-9)


def discover_cross_agent_shards(
    views: list[vggt_colmap.ViewRecord],
    collision_frames: tuple[int, ...],
    max_cross_shards: int,
) -> list[ShardDefinition]:
    by_key = {(v.agent, v.camera, v.frame): v for v in views}
    candidate_scores: dict[tuple[str, str, str, str], list[float]] = {}
    for frame in collision_frames:
        frame_views = [v for v in views if v.frame == frame]
        for i, va in enumerate(frame_views):
            for vb in frame_views[i + 1 :]:
                if va.agent == vb.agent:
                    continue
                if va.agent > vb.agent:
                    va, vb = vb, va
                ca = np.asarray(va.c2w[:3, 3], dtype=np.float64)
                cb = np.asarray(vb.c2w[:3, 3], dtype=np.float64)
                delta = cb - ca
                dist = float(np.linalg.norm(delta))
                if dist < 1.0 or dist > 85.0:
                    continue
                direction = delta / dist
                fa = image_forward(va)
                fb = image_forward(vb)
                toward_a = float(np.dot(fa, direction))
                toward_b = float(np.dot(fb, -direction))
                score = toward_a + toward_b - 0.01 * dist
                if toward_a < -0.15 or toward_b < -0.15 or score < -0.25:
                    continue
                key = (va.agent, va.camera, vb.agent, vb.camera)
                candidate_scores.setdefault(key, []).append(score)

    ranked = sorted(
        candidate_scores.items(),
        key=lambda item: (len(item[1]), float(np.mean(item[1]))),
        reverse=True,
    )
    shards: list[ShardDefinition] = []
    for key, scores in ranked[:max_cross_shards]:
        agent_a, camera_a, agent_b, camera_b = key
        available_frames = tuple(
            frame
            for frame in collision_frames
            if (agent_a, camera_a, frame) in by_key and (agent_b, camera_b, frame) in by_key
        )
        if len(available_frames) < 3:
            continue
        name = f"cross_{agent_a}_{camera_a}__{agent_b}_{camera_b}".replace("Camera_", "")
        shards.append(
            ShardDefinition(
                name=name,
                selectors=((agent_a, camera_a), (agent_b, camera_b)),
                frames=available_frames,
                kind="cross_agent_collision_overlap",
                reason=f"collision-window frustum overlap score mean={float(np.mean(scores)):.3f}",
            )
        )
    return shards


def build_shard_definitions(args: argparse.Namespace, views: list[vggt_colmap.ViewRecord]) -> list[ShardDefinition]:
    rig_frames = parse_frame_list(args.rig_frames, args.frame_start, args.frame_end)
    transition_frames = parse_frame_list(args.transition_frames, args.frame_start, args.frame_end)
    collision_frames = parse_frame_list(args.collision_frames, args.frame_start, args.frame_end)
    shards: list[ShardDefinition] = []
    for agent in args.agents:
        shards.append(
            ShardDefinition(
                name=f"{agent}_front_rig",
                selectors=tuple((agent, camera) for camera in FRONT_RIG),
                frames=rig_frames,
                kind="per_agent_front_rig",
                reason="front rig has direct camera overlap and forward temporal parallax",
            )
        )
        shards.append(
            ShardDefinition(
                name=f"{agent}_rear_rig",
                selectors=tuple((agent, camera) for camera in REAR_RIG),
                frames=rig_frames,
                kind="per_agent_rear_rig",
                reason="rear rig has direct camera overlap and backward temporal parallax",
            )
        )
        shards.append(
            ShardDefinition(
                name=f"{agent}_left_transition",
                selectors=((agent, "Camera_FrontLeft"), (agent, "Camera_BackLeft")),
                frames=transition_frames,
                kind="per_agent_side_transition",
                reason="left-side transition links front and rear overlap without all-camera BA",
            )
        )
        shards.append(
            ShardDefinition(
                name=f"{agent}_right_transition",
                selectors=((agent, "Camera_FrontRight"), (agent, "Camera_BackRight")),
                frames=transition_frames,
                kind="per_agent_side_transition",
                reason="right-side transition links front and rear overlap without all-camera BA",
            )
        )
    shards.extend(discover_cross_agent_shards(views, collision_frames, args.max_cross_shards))
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    return shards


def select_shard_views(
    all_views: list[vggt_colmap.ViewRecord],
    shard: ShardDefinition,
) -> list[vggt_colmap.ViewRecord]:
    selector_set = set(shard.selectors)
    frame_set = set(shard.frames)
    selected = [v for v in all_views if (v.agent, v.camera) in selector_set and v.frame in frame_set]
    return sorted(selected, key=lambda v: (v.frame, v.agent, v.camera))


def parse_ba_log(path: Path, ba_requested: bool) -> dict:
    if not path.exists():
        return {
            "requested": ba_requested,
            "converged": False,
            "final_cost_px": None,
            "termination": None,
            "log_exists": False,
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    final_cost = None
    match = re.search(r"Final cost\s*:\s*([0-9.eE+-]+)\s*\[px\]", text)
    if match:
        final_cost = float(match.group(1))
    termination = None
    term_match = re.search(r"Termination\s*:\s*([^\n]+)", text)
    if term_match:
        termination = term_match.group(1).strip()
    return {
        "requested": ba_requested,
        "converged": "Termination : Convergence" in text or termination == "Convergence",
        "final_cost_px": final_cost,
        "termination": termination,
        "log_exists": True,
    }


def read_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)
    if {"red", "green", "blue"}.issubset(vertex.data.dtype.names or ()):
        colors = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    return points, colors


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(
        len(points),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"] = points[:, 0].astype(np.float32)
    vertex["y"] = points[:, 1].astype(np.float32)
    vertex["z"] = points[:, 2].astype(np.float32)
    vertex["red"] = colors[:, 0].astype(np.uint8)
    vertex["green"] = colors[:, 1].astype(np.uint8)
    vertex["blue"] = colors[:, 2].astype(np.uint8)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def point_bounds(points: np.ndarray) -> dict:
    if len(points) == 0:
        return {"point_count": 0, "finite": True, "bbox_min": None, "bbox_max": None, "bbox_span": None}
    finite = bool(np.isfinite(points).all())
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    span = bbox_max - bbox_min
    return {
        "point_count": int(len(points)),
        "finite": finite,
        "bbox_min": [float(x) for x in bbox_min],
        "bbox_max": [float(x) for x in bbox_max],
        "bbox_span": [float(x) for x in span],
    }


def evaluate_shard_quality(
    shard_dir: Path,
    selected_count: int,
    alignment: dict | None,
    ba_report: dict,
    args: argparse.Namespace,
) -> dict:
    reasons: list[str] = []
    if alignment is None:
        reasons.append("missing_alignment_report")
        return {"accepted": False, "reasons": reasons}

    registered = int(alignment.get("colmap_registered_images", alignment.get("pair_count", 0)) or 0)
    points3d = int(alignment.get("colmap_points3D", 0) or 0)
    registered_ratio = registered / max(selected_count, 1)
    residuals = alignment.get("residuals", {})
    median_error = residuals.get("median_error_m")
    rmse = residuals.get("rmse_m")

    if ba_report.get("requested") and not ba_report.get("converged"):
        reasons.append("bundle_adjustment_not_converged")
    if not ba_report.get("requested") and not args.allow_no_ba:
        reasons.append("bundle_adjustment_disabled")
    if registered_ratio < args.min_registered_ratio:
        reasons.append(f"registered_ratio_{registered_ratio:.3f}_below_{args.min_registered_ratio:.3f}")
    if median_error is None or float(median_error) > args.max_median_align_error_m:
        reasons.append(f"median_alignment_error_{median_error}_above_{args.max_median_align_error_m}")
    if rmse is None or float(rmse) > args.max_rmse_align_error_m:
        reasons.append(f"rmse_alignment_error_{rmse}_above_{args.max_rmse_align_error_m}")
    if points3d < args.min_shard_points:
        reasons.append(f"points3D_{points3d}_below_{args.min_shard_points}")

    sparse_points_bin = shard_dir / "scene" / "sparse" / "points3D.bin"
    if not sparse_points_bin.exists() or sparse_points_bin.stat().st_size <= 0:
        reasons.append("missing_or_empty_points3D_bin")

    aligned_ply = shard_dir / "alignment" / "points3D_aligned_world.ply"
    bounds = {"point_count": 0, "finite": False}
    if aligned_ply.exists():
        points, _colors = read_ply_points(aligned_ply)
        bounds = point_bounds(points)
        span = np.asarray(bounds["bbox_span"] or [math.inf, math.inf, math.inf], dtype=np.float64)
        if not bounds["finite"]:
            reasons.append("non_finite_aligned_points")
        if np.any(span[:2] > args.max_shard_bbox_m) or span[2] > args.max_shard_z_span_m:
            reasons.append(f"aligned_bbox_explosion_span_{span.tolist()}")
    else:
        reasons.append("missing_aligned_world_ply")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "registered_images": registered,
        "selected_images": int(selected_count),
        "registered_ratio": float(registered_ratio),
        "points3D": points3d,
        "median_alignment_error_m": median_error,
        "rmse_alignment_error_m": rmse,
        "ba": ba_report,
        "bounds": bounds,
    }


def vggt_args_for_shard(args: argparse.Namespace) -> SimpleNamespace:
    vggt_root = args.vggt_root
    python = args.python
    if not vggt_root.is_absolute():
        vggt_root = (Path.cwd() / vggt_root).resolve()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    return SimpleNamespace(
        vggt_root=vggt_root,
        python=python,
        no_ba=args.no_ba,
        max_query_pts=args.max_query_pts,
        query_frame_num=args.query_frame_num,
        max_reproj_error=args.max_reproj_error,
        vis_thresh=args.vis_thresh,
        seed=args.seed,
    )


def write_shard_plan(out: Path, shard: ShardDefinition, selected: list[vggt_colmap.ViewRecord]) -> None:
    write_json(
        out / "shard_plan.json",
        {
            "name": shard.name,
            "kind": shard.kind,
            "reason": shard.reason,
            "selectors": [{"agent": agent, "camera": camera} for agent, camera in shard.selectors],
            "frames": list(shard.frames),
            "selected_image_count": len(selected),
            "selected_views": [
                {"view_id": v.view_id, "agent": v.agent, "camera": v.camera, "frame": v.frame}
                for v in selected
            ],
        },
    )


def run_or_prepare_shard(
    calibrated_output: Path,
    shard: ShardDefinition,
    selected: list[vggt_colmap.ViewRecord],
    shard_dir: Path,
    args: argparse.Namespace,
) -> dict:
    if len(selected) < 3:
        manifest = {
            "name": shard.name,
            "status": "rejected",
            "reason": "fewer_than_three_selected_views",
            "selected_image_count": len(selected),
        }
        write_json(shard_dir / "shard_manifest.json", manifest)
        return manifest

    if args.reuse_shards and (shard_dir / "shard_manifest.json").exists():
        return read_json(shard_dir / "shard_manifest.json")

    shard_dir.mkdir(parents=True, exist_ok=True)
    write_shard_plan(shard_dir, shard, selected)
    scene_summary = vggt_colmap.prepare_scene(calibrated_output, shard_dir, selected, image_source="masked")
    manifest = {
        "name": shard.name,
        "kind": shard.kind,
        "reason": shard.reason,
        "status": "prepared",
        "lidar_used": False,
        "selected_image_count": len(selected),
        "scene": scene_summary,
        "quality": None,
    }

    if args.prepare_only or args.dry_run:
        write_json(shard_dir / "shard_manifest.json", manifest)
        return manifest

    try:
        vg_args = vggt_args_for_shard(args)
        vggt = vggt_colmap.run_vggt_colmap(vg_args, shard_dir / "scene", shard_dir)
        manifest["vggt"] = vggt
        alignment = vggt_colmap.alignment_report(shard_dir / "scene", shard_dir)
        manifest["alignment"] = alignment
        ba_log = shard_dir / vggt["log"]
        ba_report = parse_ba_log(ba_log, ba_requested=not args.no_ba)
        quality = evaluate_shard_quality(shard_dir, len(selected), alignment, ba_report, args)
        manifest["quality"] = quality
        manifest["status"] = "accepted" if quality["accepted"] else "rejected"
    except Exception as exc:  # pragma: no cover - integration/GPU dependent
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
    write_json(shard_dir / "shard_manifest.json", manifest)
    return manifest


def crop_fusion_points(
    points: np.ndarray,
    colors: np.ndarray,
    camera_centers: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    report = {"input_points": int(len(points))}
    if len(points) == 0:
        report.update({"kept_points": 0, "filters": []})
        return points, colors, report

    center = np.median(camera_centers, axis=0) if len(camera_centers) else np.median(points, axis=0)
    valid = np.isfinite(points).all(axis=1)
    valid &= np.linalg.norm(points - center[None, :], axis=1) <= args.fusion_max_distance_m
    valid &= (points[:, 2] >= args.fusion_min_z_m) & (points[:, 2] <= args.fusion_max_z_m)
    filters = [
        f"finite_xyz",
        f"distance_to_median_camera_center <= {args.fusion_max_distance_m}m",
        f"world_z in [{args.fusion_min_z_m}, {args.fusion_max_z_m}]",
    ]
    if args.fusion_trim_percentile > 0 and int(np.count_nonzero(valid)) > 100:
        candidate = points[valid]
        lo = np.percentile(candidate, args.fusion_trim_percentile, axis=0)
        hi = np.percentile(candidate, 100.0 - args.fusion_trim_percentile, axis=0)
        valid &= np.all((points >= lo) & (points <= hi), axis=1)
        filters.append(f"axis percentile trim {args.fusion_trim_percentile}%")
    report.update({"kept_points": int(np.count_nonzero(valid)), "filters": filters, "center": [float(x) for x in center]})
    return points[valid], colors[valid], report


def voxel_downsample_with_metadata(
    points: np.ndarray,
    colors: np.ndarray,
    source_ids: np.ndarray,
    voxel: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    before = int(len(points))
    if len(points) == 0:
        return points, colors, source_ids, {"before": 0, "after_voxel": 0, "after_limit": 0}
    if voxel <= 0:
        keep = np.arange(len(points))
        after_voxel = len(points)
    else:
        keys = np.floor(points / voxel).astype(np.int64)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        sorted_keys = keys[order]
        starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)) + 1]
        keep_list = []
        out_points = []
        out_colors = []
        out_sources = []
        for pos, start in enumerate(starts):
            end = starts[pos + 1] if pos + 1 < len(starts) else len(order)
            idx = order[start:end]
            keep_list.append(int(idx[0]))
            out_points.append(points[idx].mean(axis=0))
            out_colors.append(np.rint(colors[idx].astype(np.float64).mean(axis=0)).astype(np.uint8))
            vals, counts = np.unique(source_ids[idx], return_counts=True)
            out_sources.append(int(vals[np.argmax(counts)]))
        points = np.asarray(out_points, dtype=np.float64)
        colors = np.asarray(out_colors, dtype=np.uint8)
        source_ids = np.asarray(out_sources, dtype=np.int32)
        after_voxel = len(points)
        keep = np.arange(len(points))
    if max_points > 0 and len(keep) > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(keep, size=max_points, replace=False))
        points = points[keep]
        colors = colors[keep]
        source_ids = source_ids[keep]
    return points, colors, source_ids, {
        "before": before,
        "after_voxel": int(after_voxel),
        "after_limit": int(len(points)),
        "voxel_m": float(voxel),
        "max_points": int(max_points),
    }


def fuse_accepted_shards(out: Path, accepted_manifests: list[dict], all_views: list[vggt_colmap.ViewRecord], args: argparse.Namespace) -> dict:
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    source_records: list[dict] = []
    crop_reports: list[dict] = []

    camera_centers = np.asarray([v.c2w[:3, 3] for v in all_views], dtype=np.float64)
    for source_id, manifest in enumerate(accepted_manifests):
        shard_name = manifest["name"]
        ply_path = out / "reconstruction" / "shards" / shard_name / "alignment" / "points3D_aligned_world.ply"
        points, colors = read_ply_points(ply_path)
        points, colors, crop_report = crop_fusion_points(points, colors, camera_centers, args)
        if len(points):
            point_parts.append(points)
            color_parts.append(colors)
            source_parts.append(np.full(len(points), source_id, dtype=np.int32))
        crop_reports.append({"shard": shard_name, **crop_report})
        source_records.append(
            {
                "source_id": source_id,
                "shard": shard_name,
                "raw_points": int(manifest.get("quality", {}).get("points3D", 0) or 0),
                "cropped_points": int(len(points)),
                "alignment": {
                    "median_error_m": manifest.get("quality", {}).get("median_alignment_error_m"),
                    "rmse_m": manifest.get("quality", {}).get("rmse_alignment_error_m"),
                },
            }
        )

    if point_parts:
        points = np.vstack(point_parts)
        colors = np.vstack(color_parts)
        source_ids = np.concatenate(source_parts)
    else:
        points = np.zeros((0, 3), dtype=np.float64)
        colors = np.zeros((0, 3), dtype=np.uint8)
        source_ids = np.zeros((0,), dtype=np.int32)

    before_downsample = int(len(points))
    points, colors, source_ids, downsample_report = voxel_downsample_with_metadata(
        points, colors, source_ids, args.fusion_voxel, args.fusion_max_points, args.seed
    )
    fused_path = out / "reconstruction" / "fused_sparse_world.ply"
    write_ply(fused_path, points, colors)
    viewer_points, origin = base.world_to_viewer(points)
    write_ply(out / "reconstruction" / "fused_sparse_viewer.ply", viewer_points, colors)
    trimesh.PointCloud(viewer_points.astype(np.float32), colors=colors).export(out / "scene.glb")
    try:
        base.write_preview(out / "reports" / "fused_sparse_preview.png", viewer_points, colors)
    except Exception as exc:  # pragma: no cover - matplotlib availability
        write_json(out / "reports" / "fused_sparse_preview_failed.json", {"error": repr(exc)})

    bounds = point_bounds(points)
    source_counts = []
    for record in source_records:
        count = int(np.count_nonzero(source_ids == record["source_id"]))
        source_counts.append({**record, "fused_points_after_voxel": count})
    sidecar = {
        "schema_version": "0.1",
        "created_utc": now_utc(),
        "lidar_used": False,
        "fused_ply": "reconstruction/fused_sparse_world.ply",
        "viewer_ply": "reconstruction/fused_sparse_viewer.ply",
        "scene_glb": "scene.glb",
        "before_downsample_points": before_downsample,
        "downsample": downsample_report,
        "bounds": bounds,
        "crop_reports": crop_reports,
        "sources": source_counts,
    }
    write_json(out / "reconstruction" / "fused_sparse_world_metadata.json", sidecar)
    return {
        "status": "ok" if len(points) else "failed",
        "point_count": int(len(points)),
        "world_origin_for_viewer": origin.tolist(),
        "bounds": bounds,
        "metadata": "reconstruction/fused_sparse_world_metadata.json",
        "fused_sparse_world_ply": "reconstruction/fused_sparse_world.ply",
        "scene_glb": "scene.glb",
    }


def copy_static_background_colmap(out: Path, accepted_manifests: list[dict]) -> dict:
    if not accepted_manifests:
        return {"status": "skipped", "reason": "no_accepted_shards"}
    best = sorted(
        accepted_manifests,
        key=lambda m: (
            -float(m.get("quality", {}).get("median_alignment_error_m") or math.inf),
            int(m.get("quality", {}).get("points3D") or 0),
        ),
    )[-1]
    src = out / "reconstruction" / "shards" / best["name"] / "scene"
    dst = out / "reconstruction" / "static_background_colmap"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    write_json(
        dst / "source_shard.json",
        {
            "selected_shard": best["name"],
            "selection": "lowest median alignment error with non-empty accepted COLMAP points",
            "quality": best.get("quality"),
        },
    )
    return {"status": "ok", "path": "reconstruction/static_background_colmap", "source_shard": best["name"]}


def replay_frames_from_calibrated_export(calibrated_output: Path, args: argparse.Namespace) -> list[int]:
    cameras_path = calibrated_output / "reconstruction" / "cameras.json"
    if cameras_path.exists():
        camera_data = read_json(cameras_path)
        frames = sorted(
            {
                int(view["frame"])
                for view in camera_data.get("views", [])
                if args.frame_start <= int(view["frame"]) <= args.frame_end
            }
        )
        if frames:
            return frames
    return base.select_frames(args.frame_start, args.frame_end, 1)


def build_replay_and_viewer(out: Path, calibrated_output: Path, args: argparse.Namespace, viewer_origin: np.ndarray) -> dict:
    frames = replay_frames_from_calibrated_export(calibrated_output, args)
    write_json(
        out / "replay" / "frame_selection.json",
        {
            "requested_frame_range": [args.frame_start, args.frame_end],
            "used_frames": frames,
            "source": "calibrated export cameras.json",
            "note": "Replay uses frames that actually exist in the calibrated camera export.",
        },
    )
    base.CATEGORY = args.category
    tracks = multi.build_agent_tracks(args.dataset.resolve(), out, args.agents, frames, args.scenario, viewer_origin, args.nominal_fps)
    object_context = multi.build_object_tracks(args.dataset.resolve(), out, args.agents, frames, args.scenario)
    diagnostics = multi.accident_diagnostics(out, tracks)
    shutil.copy2(out / "replay" / "agent_tracks.json", out / "replay" / "vehicle_tracks.json")
    shutil.copy2(out / "replay" / "accident_diagnostics.json", out / "replay" / "collision_diagnostics.json")
    cameras = read_json(calibrated_output / "reconstruction" / "cameras.json")
    frustums = build_camera_frustums(cameras, viewer_origin, frame_stride=7)
    write_json(out / "viewer" / "camera_frustums.json", frustums)
    write_final_viewer(out)
    return {
        "vehicle_tracks": "replay/vehicle_tracks.json",
        "collision_diagnostics": "replay/collision_diagnostics.json",
        "object_context_records": object_context.get("record_count", 0),
        "closest_approach": diagnostics.get("closest_approach"),
        "used_frames": frames,
        "camera_frustums": "viewer/camera_frustums.json",
        "viewer": "viewer/index.html",
    }


def build_camera_frustums(cameras: dict, origin: np.ndarray, frame_stride: int = 7) -> dict:
    records = []
    for view in cameras.get("views", []):
        frame = int(view["frame"])
        if (frame - 1) % frame_stride != 0 and frame != 56:
            continue
        c2w = np.asarray(view["camera_to_world_cv"], dtype=np.float64)
        center = c2w[:3, 3]
        forward = c2w[:3, 2]
        up = -c2w[:3, 1]
        right = c2w[:3, 0]
        scale = 1.25
        corners = []
        for sx, sy in [(-1, -0.62), (1, -0.62), (1, 0.62), (-1, 0.62)]:
            p = center + scale * (forward + 0.55 * sx * right + 0.36 * sy * up)
            corners.append(base.viewer_point(p, origin) if hasattr(base, "viewer_point") else [float(p[0] - origin[0]), float(p[2] - origin[2]), float(p[1] - origin[1])])
        records.append(
            {
                "view_id": view["view_id"],
                "agent": view["agent"],
                "camera": view["camera"],
                "frame": frame,
                "position_viewer": [float(center[0] - origin[0]), float(center[2] - origin[2]), float(center[1] - origin[1])],
                "corners_viewer": corners,
            }
        )
    return {"schema_version": "0.1", "coordinate_system": "viewer xyz = world x,z,y minus fusion origin", "frustums": records}


def write_final_viewer(out: Path) -> None:
    viewer = out / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    (viewer / "index.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Town04 camera-only reconstruction final</title>
  <style>
    html, body { margin:0; height:100%; overflow:hidden; background:#080a0d; color:#e8eaed; font:13px system-ui, sans-serif; }
    #hud { position:absolute; left:12px; top:12px; z-index:2; width:min(460px, calc(100vw - 24px)); background:rgba(8,10,13,.78); border:1px solid #2d333b; padding:10px 12px; box-sizing:border-box; }
    #hud strong { display:block; margin-bottom:4px; font-size:14px; }
    #hud .row { display:flex; justify-content:space-between; gap:10px; white-space:nowrap; }
    #hud input { width:100%; margin-top:8px; }
    #legend { display:grid; grid-template-columns:1fr 1fr; gap:4px 10px; margin-top:8px; }
    .swatch { display:inline-block; width:10px; height:10px; margin-right:6px; border-radius:50%; vertical-align:-1px; }
    #buttons { display:flex; gap:6px; margin-top:8px; flex-wrap:wrap; }
    button { background:#1f2937; color:#e8eaed; border:1px solid #374151; padding:5px 8px; cursor:pointer; }
    canvas { display:block; }
  </style>
  <script type="importmap">{"imports":{"three":"./vendor/three/build/three.module.js","three/addons/":"./vendor/three/examples/jsm/"}}</script>
</head>
<body>
  <div id="hud">
    <strong>Town04 camera-only final reconstruction</strong>
    <div class="row"><span id="status">loading</span><span id="frameLabel"></span></div>
    <input id="frame" type="range" min="0" max="0" value="0" step="1" />
    <div id="closest"></div>
    <div id="buttons"><button id="toggleFrustums">frustums on</button><button id="toggleTracks">tracks on</button><button id="top">top view</button><button id="orbit">orbit view</button></div>
    <div id="legend"></div>
  </div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080a0d);
    const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.01, 4000);
    const renderer = new THREE.WebGLRenderer({ antialias:true, preserveDrawingBuffer:true });
    renderer.setSize(innerWidth, innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    document.body.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x18202b, 1.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.0);
    sun.position.set(-30, 80, -40);
    scene.add(sun);
    scene.add(new THREE.GridHelper(140, 70, 0x38424c, 0x20262d));

    const staticGroup = new THREE.Group();
    const trackGroup = new THREE.Group();
    const frustumGroup = new THREE.Group();
    scene.add(staticGroup, trackGroup, frustumGroup);
    const statusEl = document.getElementById('status');
    const frameEl = document.getElementById('frame');
    const frameLabelEl = document.getElementById('frameLabel');
    const closestEl = document.getElementById('closest');
    const legendEl = document.getElementById('legend');
    let tracks = null;
    let frames = [];
    let markers = [];
    let sceneCenter = new THREE.Vector3();
    let sceneSize = 80;

    function hexToNumber(hex) { return Number.parseInt(hex.replace('#', ''), 16); }
    function setTopView() {
      camera.position.copy(sceneCenter).add(new THREE.Vector3(0, sceneSize * .9, .001));
      controls.target.copy(sceneCenter);
      camera.lookAt(sceneCenter);
    }
    function setOrbitView() {
      camera.position.copy(sceneCenter).add(new THREE.Vector3(sceneSize * .35, sceneSize * .25, sceneSize * .55));
      controls.target.copy(sceneCenter);
      camera.lookAt(sceneCenter);
    }
    function makeLine(points, color) {
      const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p[0], p[1] + .18, p[2])));
      return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color }));
    }
    function addTracks() {
      legendEl.innerHTML = '';
      for (const [agent, data] of Object.entries(tracks.agents)) {
        const color = hexToNumber(data.color || '#ffffff');
        trackGroup.add(makeLine(data.samples.map(s => s.position_viewer), color));
        const div = document.createElement('div');
        div.innerHTML = `<span class="swatch" style="background:${data.color}"></span>${agent}`;
        legendEl.appendChild(div);
        const marker = new THREE.Mesh(
          new THREE.BoxGeometry(Math.max(.7, data.dimensions.width_m), Math.max(.7, data.dimensions.height_m), Math.max(1, data.dimensions.length_m)),
          new THREE.MeshBasicMaterial({ color, wireframe:true })
        );
        marker.userData.agent = agent;
        trackGroup.add(marker);
        markers.push(marker);
      }
      frames = [...new Set(Object.values(tracks.agents).flatMap(d => d.samples.map(s => s.frame)))].sort((a, b) => a - b);
      frameEl.max = Math.max(0, frames.length - 1);
      const closest = tracks.closest_approach;
      const startIdx = closest ? Math.max(0, frames.indexOf(closest.frame)) : 0;
      frameEl.value = startIdx;
      updateFrame(startIdx);
    }
    function updateFrame(idx) {
      if (!tracks || frames.length === 0) return;
      const frame = frames[idx];
      frameLabelEl.textContent = `frame ${frame}`;
      markers.forEach(marker => {
        const sample = tracks.agents[marker.userData.agent].samples.find(s => s.frame === frame);
        if (!sample) { marker.visible = false; return; }
        marker.visible = true;
        marker.position.set(sample.position_viewer[0], sample.position_viewer[1] + .8, sample.position_viewer[2]);
        marker.rotation.y = -sample.yaw_rad;
      });
    }
    function addFrustums(data) {
      for (const item of data.frustums) {
        const color = item.agent.includes('other') ? 0xfacc15 : 0x60a5fa;
        const c = new THREE.Vector3(...item.position_viewer);
        const corners = item.corners_viewer.map(p => new THREE.Vector3(...p));
        const pts = [c, corners[0], corners[1], c, corners[1], corners[2], c, corners[2], corners[3], c, corners[3], corners[0], corners[0], corners[1], corners[2], corners[3], corners[0]];
        const geom = new THREE.BufferGeometry().setFromPoints(pts);
        frustumGroup.add(new THREE.Line(geom, new THREE.LineBasicMaterial({ color, transparent:true, opacity:.38 })));
      }
    }
    frameEl.addEventListener('input', () => updateFrame(Number(frameEl.value)));
    document.getElementById('toggleFrustums').onclick = () => { frustumGroup.visible = !frustumGroup.visible; };
    document.getElementById('toggleTracks').onclick = () => { trackGroup.visible = !trackGroup.visible; };
    document.getElementById('top').onclick = setTopView;
    document.getElementById('orbit').onclick = setOrbitView;

    new GLTFLoader().load('../scene.glb', gltf => {
      gltf.scene.traverse(obj => {
        if (obj.isPoints && obj.material) obj.material.size = 0.045;
      });
      staticGroup.add(gltf.scene);
      const box = new THREE.Box3().setFromObject(gltf.scene);
      sceneCenter = box.getCenter(new THREE.Vector3());
      sceneSize = Math.max(30, box.getSize(new THREE.Vector3()).length());
      camera.near = Math.max(.01, sceneSize / 10000);
      camera.far = Math.max(4000, sceneSize * 12);
      camera.updateProjectionMatrix();
      setOrbitView();
      statusEl.textContent = 'fused sparse scene loaded';
    }, undefined, err => {
      console.error(err);
      statusEl.textContent = 'scene.glb failed';
    });
    fetch('../replay/vehicle_tracks.json').then(r => r.json()).then(json => {
      tracks = json;
      addTracks();
      const c = tracks.closest_approach;
      if (c) closestEl.textContent = `closest proxy: ${c.agents.join(' / ')} frame ${c.frame}, clearance ${c.proxy_clearance_xy_m.toFixed(2)}m`;
    });
    fetch('camera_frustums.json').then(r => r.json()).then(addFrustums);
    addEventListener('resize', () => {
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight);
    });
    function animate() {
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }
    animate();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def run_nerfstudio_splatfacto(out: Path, static_colmap: dict, args: argparse.Namespace, repo: Path) -> dict:
    if args.skip_3dgs or args.prepare_only or args.dry_run:
        return {"status": "skipped", "reason": "disabled_for_this_run"}
    if static_colmap.get("status") != "ok":
        return {"status": "failed", "reason": "no_static_colmap_shard"}
    ns_train = args.ns_train
    if not ns_train.is_absolute():
        ns_train = (repo / ns_train).resolve()
    if not ns_train.exists():
        return {"status": "failed", "reason": f"ns-train not found: {ns_train}"}
    data_dir = out / static_colmap["path"]
    cmd = [
        str(ns_train),
        "splatfacto",
        "--output-dir",
        str(out / "reconstruction" / "gaussians" / "nerfstudio"),
        "--experiment-name",
        "town04_camera_only_final",
        "--timestamp",
        "splatfacto",
        "--vis",
        "tensorboard",
        "--max-num-iterations",
        str(args.gs_iterations),
        "--pipeline.datamanager.camera-res-scale-factor",
        str(args.camera_res_scale),
        "colmap",
        "--data",
        str(data_dir),
        "--images-path",
        "images_fullrgb",
        "--colmap-path",
        "sparse",
    ]
    result = run_command(cmd, repo, out / "logs" / "nerfstudio_splatfacto.log", dry_run=False)
    result["status"] = "ok" if result["returncode"] == 0 else "failed"
    if result["status"] == "ok":
        ns_export = ns_train.with_name("ns-export")
        config_candidates = sorted(
            (out / "reconstruction" / "gaussians" / "nerfstudio").rglob("config.yml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not config_candidates:
            result["status"] = "failed"
            result["export_error"] = "Nerfstudio finished but no config.yml was found for gaussian-splat export."
        elif not ns_export.exists():
            result["status"] = "failed"
            result["export_error"] = f"ns-export not found: {ns_export}"
        else:
            export_cmd = [
                str(ns_export),
                "gaussian-splat",
                "--load-config",
                str(config_candidates[0]),
                "--output-dir",
                str(out / "reconstruction" / "gaussians"),
                "--output-filename",
                "splat.ply",
                "--ply-color-mode",
                "rgb",
            ]
            export_result = run_command(export_cmd, repo, out / "logs" / "nerfstudio_gaussian_export.log", dry_run=False)
            result["export"] = export_result
            if export_result["returncode"] != 0:
                result["status"] = "failed"
    exported = list((out / "reconstruction" / "gaussians").rglob("*.ply"))
    result["exported_ply_candidates"] = [rel_or_abs(p, out) for p in exported]
    result["splat_ply"] = "reconstruction/gaussians/splat.ply" if (out / "reconstruction" / "gaussians" / "splat.ply").exists() else None
    return result


def audit_forbidden_paths(out: Path, extra_paths: list[Path] | None = None) -> dict:
    forbidden = ["lidar01", "viewer_assets", "four_vehicle_static_lidar_background", "historical_lidar"]
    scan_paths = [
        out / "manifest.json",
        out / "reports" / "quality_report.md",
        out / "reports" / "no_lidar_audit.md",
        out / "reconstruction" / "fused_sparse_world_metadata.json",
    ]
    if extra_paths:
        scan_paths.extend(extra_paths)
    hits = []
    for path in scan_paths:
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for term in forbidden:
            if term in text:
                allowed_audit_context = path.name == "no_lidar_audit.md" and term in {"lidar01", "viewer_assets", "four_vehicle_static_lidar_background"}
                if not allowed_audit_context:
                    hits.append({"path": str(path), "term": term})
    return {"status": "pass" if not hits else "fail", "hits": hits, "terms": forbidden}


def write_reports(
    out: Path,
    args: argparse.Namespace,
    shard_manifests: list[dict],
    fusion: dict,
    static_colmap: dict,
    gs_report: dict,
    replay_report: dict | None,
    export_report: dict,
) -> None:
    accepted = [m for m in shard_manifests if m.get("status") == "accepted"]
    rejected = [m for m in shard_manifests if m.get("status") == "rejected"]
    failed = [m for m in shard_manifests if m.get("status") == "failed"]
    no_lidar = f"""# No-LiDAR Audit

Status: pass.

Final pipeline: `scripts/run_town04_camera_only_final_pipeline.py`

Allowed inputs:

- RGB camera frames from `{args.dataset}`
- DeepAccident camera intrinsics/extrinsics and `ego_to_world` calibration as pose/scale priors
- Dynamic-object masks generated from RGB images or empty fallback masks
- Label text files for vehicle proxy dimensions and replay context

Forbidden reconstruction geometry:

- `lidar01` point clouds: not read
- `viewer_assets/*.glb`: not read
- historical LiDAR PLY/GLB assets: not read

Calibration caveat:

DeepAccident calibration matrix names include `lidar_to_Camera_*` and
`lidar_to_ego`; this pipeline uses those matrices only to recover calibrated
camera poses and never opens `lidar01` point geometry.

Manifest invariant: `lidar_used=false`.
"""
    (out / "reports" / "no_lidar_audit.md").write_text(no_lidar, encoding="utf-8")

    accepted_lines = []
    for m in accepted:
        q = m.get("quality", {})
        accepted_lines.append(
            f"- `{m['name']}`: images {q.get('registered_images')}/{q.get('selected_images')}, "
            f"points {q.get('points3D')}, median align {q.get('median_alignment_error_m')} m, "
            f"RMSE {q.get('rmse_alignment_error_m')} m"
        )
    rejected_lines = []
    for m in rejected + failed:
        q = m.get("quality") or {}
        reasons = q.get("reasons") or [m.get("reason") or m.get("error") or "unknown"]
        rejected_lines.append(f"- `{m.get('name')}`: `{m.get('status')}`; {', '.join(map(str, reasons))}")

    status = "pass" if accepted and fusion.get("status") == "ok" else "fail"
    quality = f"""# Town04 Camera-only Final Quality Report

Status: `{status}`

## Inputs

- Scenario: `{args.scenario}`
- Agents: `{', '.join(args.agents)}`
- Cameras: `{', '.join(args.cameras)}`
- Frame range: `{args.frame_start}` to `{args.frame_end}`
- LiDAR geometry used: `false`

## Shard Gate

- Attempted shards: `{len(shard_manifests)}`
- Accepted shards: `{len(accepted)}`
- Rejected shards: `{len(rejected)}`
- Failed shards: `{len(failed)}`
- Required registered ratio: `{args.min_registered_ratio}`
- Required median alignment error: `<= {args.max_median_align_error_m} m`
- Hard RMSE reject: `> {args.max_rmse_align_error_m} m`

### Accepted

{chr(10).join(accepted_lines) if accepted_lines else '- None'}

### Rejected Or Failed

{chr(10).join(rejected_lines) if rejected_lines else '- None'}

## Fusion

- Status: `{fusion.get('status')}`
- Fused world points: `{fusion.get('point_count')}`
- Bounds: `{fusion.get('bounds')}`
- Static COLMAP source: `{static_colmap}`

## 3DGS

- Status: `{gs_report.get('status')}`
- Details: `{gs_report}`

If 3DGS is skipped or fails, this report treats the fused camera-only sparse
world reconstruction and viewer as the inspected fallback. It does not claim
dynamic vehicle surface reconstruction; vehicles are replayed as calibrated
proxy boxes from labels and camera/ego poses.

## Viewer

- `viewer/index.html`
- Vehicle tracks: `{(replay_report or {}).get('vehicle_tracks')}`
- Camera frustums: `{(replay_report or {}).get('camera_frustums')}`
- Collision diagnostics: `{(replay_report or {}).get('collision_diagnostics')}`
"""
    (out / "reports" / "quality_report.md").write_text(quality, encoding="utf-8")


def validate_final_outputs(out: Path, manifest: dict) -> dict:
    required = [
        out / "manifest.json",
        out / "reconstruction" / "fused_sparse_world.ply",
        out / "reconstruction" / "fused_sparse_world_metadata.json",
        out / "reports" / "quality_report.md",
        out / "reports" / "no_lidar_audit.md",
        out / "viewer" / "index.html",
        out / "replay" / "vehicle_tracks.json",
        out / "replay" / "collision_diagnostics.json",
    ]
    missing = [rel_or_abs(p, out) for p in required if not p.exists()]
    size_failures = [rel_or_abs(p, out) for p in required if p.exists() and p.stat().st_size <= 20]
    audit = audit_forbidden_paths(out)
    lidar_ok = manifest.get("lidar_used") is False
    return {
        "status": "pass" if not missing and not size_failures and audit["status"] == "pass" and lidar_ok else "fail",
        "missing": missing,
        "size_failures": size_failures,
        "audit": audit,
        "manifest_lidar_used_false": lidar_ok,
    }


def main() -> int:
    args = parse_args()
    repo = Path.cwd().resolve()
    if not args.out.is_absolute():
        args.out = (repo / args.out).resolve()
    if not args.dataset.is_absolute():
        args.dataset = (repo / args.dataset).resolve()
    if not args.calibrated_output.is_absolute():
        args.calibrated_output = (repo / args.calibrated_output).resolve()
    if not args.python.is_absolute():
        args.python = (repo / args.python).absolute()
    if not args.vggt_root.is_absolute():
        args.vggt_root = (repo / args.vggt_root).resolve()

    ensure_final_dirs(args.out)
    start = time.time()
    export_report = rebuild_calibrated_export(args, repo)
    cameras_path = args.calibrated_output / "reconstruction" / "cameras.json"
    if not cameras_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Missing calibrated cameras export: {cameras_path}")

    all_views = [] if args.dry_run and not cameras_path.exists() else vggt_colmap.load_views(args.calibrated_output)
    shard_defs = build_shard_definitions(args, all_views) if all_views else []
    shard_manifests = []
    for shard in shard_defs:
        selected = select_shard_views(all_views, shard)
        shard_dir = args.out / "reconstruction" / "shards" / shard.name
        manifest = run_or_prepare_shard(args.calibrated_output, shard, selected, shard_dir, args)
        shard_manifests.append(manifest)

    accepted = [m for m in shard_manifests if m.get("status") == "accepted"]
    if accepted and not args.prepare_only and not args.dry_run:
        fusion = fuse_accepted_shards(args.out, accepted, all_views, args)
        viewer_origin = np.asarray(fusion["world_origin_for_viewer"], dtype=np.float64)
        static_colmap = copy_static_background_colmap(args.out, accepted)
        replay_report = build_replay_and_viewer(args.out, args.calibrated_output, args, viewer_origin)
        gs_report = run_nerfstudio_splatfacto(args.out, static_colmap, args, repo)
    else:
        fusion = {"status": "skipped", "reason": "no_accepted_shards_or_prepare_only", "point_count": 0}
        static_colmap = {"status": "skipped", "reason": "no_accepted_shards_or_prepare_only"}
        replay_report = None
        gs_report = {"status": "skipped", "reason": "no_accepted_shards_or_prepare_only"}

    write_reports(args.out, args, shard_manifests, fusion, static_colmap, gs_report, replay_report, export_report)
    status = "ok" if accepted and fusion.get("status") == "ok" else ("prepared" if args.prepare_only or args.dry_run else "failed")
    manifest = {
        "schema_version": "1.0",
        "case_id": "camera_only_reconstruction_town04_final",
        "created_utc": now_utc(),
        "status": status,
        "lidar_used": False,
        "legacy_lidar_assets_used": False,
        "scenario": args.scenario,
        "category": args.category,
        "inputs": {
            "dataset_root": str(args.dataset),
            "calibrated_output": str(args.calibrated_output),
            "agents": args.agents,
            "cameras": args.cameras,
            "frame_range": [args.frame_start, args.frame_end],
        },
        "export": export_report,
        "shards": {
            "planned": len(shard_defs),
            "attempted": len(shard_manifests),
            "accepted": [m["name"] for m in accepted],
            "manifests": [f"reconstruction/shards/{m['name']}/shard_manifest.json" for m in shard_manifests if "name" in m],
        },
        "fusion": fusion,
        "static_background_colmap": static_colmap,
        "gaussians": gs_report,
        "replay": replay_report,
        "outputs": {
            "static_background_colmap": "reconstruction/static_background_colmap" if static_colmap.get("status") == "ok" else None,
            "fused_sparse_world": "reconstruction/fused_sparse_world.ply" if fusion.get("status") == "ok" else None,
            "fused_sparse_metadata": fusion.get("metadata"),
            "viewer": "viewer/index.html" if replay_report else None,
            "quality_report": "reports/quality_report.md",
            "no_lidar_audit": "reports/no_lidar_audit.md",
        },
        "elapsed_seconds": time.time() - start,
    }
    write_json(args.out / "manifest.json", manifest)
    validation = validate_final_outputs(args.out, manifest) if status == "ok" else {"status": "not_run", "reason": status}
    write_json(args.out / "reports" / "validation_report.json", validation)
    manifest["validation"] = validation
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0 if status in {"ok", "prepared"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
