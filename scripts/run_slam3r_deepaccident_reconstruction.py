#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
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
CAMERAS = [
    "Camera_Front",
    "Camera_FrontLeft",
    "Camera_FrontRight",
    "Camera_Back",
    "Camera_BackLeft",
    "Camera_BackRight",
]


@dataclass(frozen=True)
class Stream:
    agent: str
    camera: str
    frames: tuple[int, ...]
    input_dir: Path
    test_name: str
    gpu_id: int

    @property
    def key(self) -> str:
        return f"{self.agent}/{self.camera}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SLAM3R over DeepAccident Town04 multi-agent camera streams and fuse them in world coordinates."
    )
    parser.add_argument("--dataset", type=Path, default=Path("deepaccident_mini_dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs/town04_type1_subtype2_slam3r_reconstruction"))
    parser.add_argument("--slam3r-root", type=Path, default=Path("third_party/SLAM3R"))
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--agents", nargs="+", default=AGENTS)
    parser.add_argument("--cameras", nargs="+", default=CAMERAS)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=49)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--skip-slam3r", action="store_true", help="Reuse existing SLAM3R per-stream outputs.")
    parser.add_argument("--resume-slam3r", action="store_true", help="Reuse completed per-stream SLAM3R preds and rerun only missing streams.")
    parser.add_argument("--static-gpu-assignment", action="store_true", help="Use round-robin GPU assignment instead of live GPU load selection.")
    parser.add_argument("--min-gpu-free-mb", type=int, default=12000)
    parser.add_argument("--max-gpu-util", type=int, default=80)
    parser.add_argument("--gpu-poll-seconds", type=float, default=15.0)
    parser.add_argument("--gpu-max-wait-seconds", type=float, default=300.0)
    parser.add_argument("--keyframe-stride", type=int, default=3)
    parser.add_argument("--win-r", type=int, default=5)
    parser.add_argument("--initial-winsize", type=int, default=5)
    parser.add_argument("--num-scene-frame", type=int, default=10)
    parser.add_argument("--max-num-register", type=int, default=10)
    parser.add_argument("--conf-thres-i2p", type=float, default=1.5)
    parser.add_argument("--conf-thres-l2w", type=float, default=10.0)
    parser.add_argument("--num-points-save", type=int, default=120000)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--buffer-strategy", choices=["reservoir", "fifo"], default="fifo")
    parser.add_argument("--points-per-stream", type=int, default=60000)
    parser.add_argument("--max-points", type=int, default=1000000)
    parser.add_argument("--point-conf-percentile", type=float, default=62.0)
    parser.add_argument("--pose-conf-percentile", type=float, default=70.0)
    parser.add_argument(
        "--fusion-mode",
        choices=["camera-ray-depth", "stream-sim3"],
        default="camera-ray-depth",
        help=(
            "camera-ray-depth keeps SLAM3R per-pixel depth in each camera frame and places it with "
            "DeepAccident calibration poses. stream-sim3 applies one Sim(3) to each whole SLAM3R stream."
        ),
    )
    parser.add_argument(
        "--mask-export",
        type=Path,
        default=Path("outputs/town04_type1_subtype2_multicam_export"),
        help="Optional camera-only export containing masks/<agent>/<camera>/frame_###.png.",
    )
    parser.add_argument("--disable-mask-filter", action="store_true")
    parser.add_argument(
        "--require-mask-filter",
        action="store_true",
        help="Skip frames that do not have a dynamic-object mask in --mask-export.",
    )
    parser.add_argument("--min-camera-depth", type=float, default=0.05)
    parser.add_argument("--max-camera-depth", type=float, default=4.5)
    parser.add_argument("--max-metric-depth-m", type=float, default=85.0)
    parser.add_argument("--depth-trim-percentile", type=float, default=1.0)
    parser.add_argument("--disable-sky-filter", action="store_true")
    parser.add_argument("--max-stream-rmse-m", type=float, default=2.5)
    parser.add_argument("--max-stream-median-error-m", type=float, default=1.8)
    parser.add_argument("--max-stream-max-error-m", type=float, default=7.0)
    parser.add_argument("--max-frame-alignment-error-m", type=float, default=4.0)
    parser.add_argument("--min-world-z", type=float, default=-2.5)
    parser.add_argument("--max-world-z", type=float, default=25.0)
    parser.add_argument("--trim-percentile", type=float, default=1.0)
    parser.add_argument("--voxel", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--nominal-fps", type=float, default=20.0)
    return parser.parse_args()


def select_existing_frames(dataset: Path, agent: str, camera: str, scenario: str, frames: list[int]) -> tuple[int, ...]:
    source_dir = base.dataset_path(dataset, agent, camera, scenario)
    existing = []
    for frame in frames:
        if (source_dir / f"{scenario}_{frame:03d}.jpg").exists():
            existing.append(frame)
    return tuple(existing)


def clean_output(out: Path, keep_slam3r: bool) -> None:
    if not keep_slam3r:
        for rel in ["input_streams", "slam3r_results"]:
            target = out / rel
            if target.exists():
                shutil.rmtree(target)
    for rel in ["reconstruction", "reports", "replay", "viewer"]:
        target = out / rel
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    for rel in ["scene.glb", "summary.json", "manifest.json"]:
        target = out / rel
        if target.exists():
            target.unlink()


def export_stream_inputs(dataset: Path, out: Path, args: argparse.Namespace) -> list[Stream]:
    requested_frames = base.select_frames(args.frame_start, args.frame_end, args.frame_step, include_end=True)
    streams: list[Stream] = []
    for agent in args.agents:
        for camera in args.cameras:
            frames = select_existing_frames(dataset, agent, camera, args.scenario, requested_frames)
            if not frames:
                continue
            input_dir = out / "input_streams" / agent / camera
            input_dir.mkdir(parents=True, exist_ok=True)
            source_dir = base.dataset_path(dataset, agent, camera, args.scenario)
            for ordinal, frame in enumerate(frames, start=1):
                src = (source_dir / f"{args.scenario}_{frame:03d}.jpg").resolve()
                dst = input_dir / f"frame_{ordinal:03d}.jpg"
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.symlink(src, dst)
            streams.append(
                Stream(
                    agent=agent,
                    camera=camera,
                    frames=frames,
                    input_dir=input_dir,
                    test_name=f"{agent}__{camera}",
                    gpu_id=args.gpus[len(streams) % len(args.gpus)],
                )
            )
    if not streams:
        raise FileNotFoundError("No DeepAccident RGB streams were exported for the requested scenario.")
    return streams


def run_one_slam3r(stream: Stream, args: argparse.Namespace, repo: Path) -> dict:
    if args.resume_slam3r and stream_output_complete(args.out, stream):
        return {
            "agent": stream.agent,
            "camera": stream.camera,
            "test_name": stream.test_name,
            "gpu_id": None,
            "frame_count": len(stream.frames),
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "cmd": [],
            "stdout_tail": "reused completed per-stream SLAM3R preds",
            "stderr_tail": "",
            "skipped_existing": True,
        }

    slam3r_root = args.slam3r_root.resolve()
    save_dir = (args.out / "slam3r_results").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(slam3r_root) + os.pathsep + env.get("PYTHONPATH", "")
    gpu_id = stream.gpu_id if args.static_gpu_assignment else wait_for_gpu(args)
    cmd = [
        sys.executable,
        str(slam3r_root / "recon.py"),
        "--test_name",
        stream.test_name,
        "--dataset",
        str(stream.input_dir.resolve()),
        "--save_dir",
        str(save_dir),
        "--gpu_id",
        str(gpu_id),
        "--keyframe_stride",
        str(args.keyframe_stride),
        "--win_r",
        str(args.win_r),
        "--num_scene_frame",
        str(args.num_scene_frame),
        "--initial_winsize",
        str(args.initial_winsize),
        "--conf_thres_i2p",
        str(args.conf_thres_i2p),
        "--conf_thres_l2w",
        str(args.conf_thres_l2w),
        "--num_points_save",
        str(args.num_points_save),
        "--update_buffer_intv",
        "1",
        "--buffer_size",
        str(args.buffer_size),
        "--buffer_strategy",
        args.buffer_strategy,
        "--max_num_register",
        str(args.max_num_register),
        "--save_preds",
    ]
    start = time.time()
    proc = subprocess.run(cmd, cwd=repo, env=env, text=True, capture_output=True)
    return {
        "agent": stream.agent,
        "camera": stream.camera,
        "test_name": stream.test_name,
        "gpu_id": gpu_id,
        "frame_count": len(stream.frames),
        "returncode": proc.returncode,
        "elapsed_seconds": time.time() - start,
        "cmd": cmd,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def stream_output_complete(out: Path, stream: Stream) -> bool:
    preds_dir = out / "slam3r_results" / stream.test_name / "preds"
    required = [
        "local_pcds.npy",
        "registered_pcds.npy",
        "local_confs.npy",
        "registered_confs.npy",
        "input_imgs.npy",
        "metadata.json",
    ]
    return all((preds_dir / name).exists() for name in required)


def query_gpu_status() -> list[dict]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return []
    status = []
    for row in proc.stdout.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 3:
            continue
        try:
            status.append(
                {
                    "index": int(parts[0]),
                    "free_mb": int(parts[1]),
                    "util_percent": int(parts[2]),
                }
            )
        except ValueError:
            continue
    return status


def wait_for_gpu(args: argparse.Namespace) -> int:
    allowed = set(args.gpus)
    start = time.time()
    last_status: list[dict] = []
    while True:
        last_status = [item for item in query_gpu_status() if item["index"] in allowed]
        eligible = [
            item
            for item in last_status
            if item["free_mb"] >= args.min_gpu_free_mb and item["util_percent"] <= args.max_gpu_util
        ]
        if eligible:
            eligible.sort(key=lambda item: (-item["free_mb"], item["util_percent"], item["index"]))
            return int(eligible[0]["index"])
        waited = time.time() - start
        if args.gpu_max_wait_seconds > 0 and waited >= args.gpu_max_wait_seconds and last_status:
            last_status.sort(key=lambda item: (-item["free_mb"], item["util_percent"], item["index"]))
            return int(last_status[0]["index"])
        time.sleep(max(1.0, float(args.gpu_poll_seconds)))


def run_slam3r_streams(streams: list[Stream], args: argparse.Namespace, repo: Path) -> list[dict]:
    if args.skip_slam3r:
        return [
            {
                "agent": stream.agent,
                "camera": stream.camera,
                "test_name": stream.test_name,
                "gpu_id": stream.gpu_id,
                "frame_count": len(stream.frames),
                "returncode": 0,
                "elapsed_seconds": 0.0,
                "cmd": [],
                "stdout_tail": "reused existing outputs",
                "stderr_tail": "",
            }
            for stream in streams
        ]
    results: list[dict] = []
    max_workers = max(1, int(args.parallel))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one_slam3r, stream, args, repo) for stream in streams]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "stream": f"{result['agent']}/{result['camera']}",
                        "returncode": result["returncode"],
                        "gpu_id": result.get("gpu_id"),
                        "skipped_existing": bool(result.get("skipped_existing")),
                        "elapsed_seconds": round(result["elapsed_seconds"], 2),
                    }
                ),
                flush=True,
            )
    results.sort(key=lambda item: (item["agent"], item["camera"]))
    failed = [item for item in results if item["returncode"] != 0]
    if failed:
        raise RuntimeError(f"SLAM3R failed for {len(failed)} streams. See reconstruction/stream_runs.json.")
    return results


def fit_similarity_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 3:
        raise ValueError("Need at least three camera centers for Sim(3) alignment.")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / len(src)
    u, svals, vt = np.linalg.svd(cov)
    d = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1.0
    rot = u @ d @ vt
    var_src = float(np.mean(np.sum(src_c * src_c, axis=1)))
    scale = float(np.trace(np.diag(svals) @ d) / max(var_src, 1e-12))
    trans = mu_dst - scale * (rot @ mu_src)
    aligned = (scale * (rot @ src.T)).T + trans
    errors = np.linalg.norm(aligned - dst, axis=1)
    return scale, rot, trans, {
        "camera_count": int(len(src)),
        "rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "median_error_m": float(np.median(errors)),
        "max_error_m": float(np.max(errors)),
        "errors_m": [float(x) for x in errors],
    }


def apply_similarity(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3).astype(np.float64)
    aligned = (scale * (rot @ flat.T)).T + trans
    return aligned.reshape(points.shape).astype(np.float32)


def estimate_intrinsic(local_pcds: np.ndarray, slam3r_root: Path) -> np.ndarray:
    sys.path.insert(0, str(slam3r_root.resolve()))
    from slam3r.utils.recon_utils import estimate_focal_knowing_depth

    height, width = local_pcds.shape[1:3]
    principal_point = torch.tensor((width / 2.0, height / 2.0), dtype=torch.float32)
    sample = torch.from_numpy(local_pcds[: min(5, len(local_pcds))]).float()
    focal = estimate_focal_knowing_depth(sample, principal_point, focal_mode="weiszfeld")
    focal_value = float(torch.nanmedian(focal).item())
    if not math.isfinite(focal_value) or focal_value <= 1.0:
        focal_value = float(max(width, height))
    intrinsic = np.eye(3, dtype=np.float64)
    intrinsic[0, 0] = focal_value
    intrinsic[1, 1] = focal_value
    intrinsic[0, 2] = width / 2.0
    intrinsic[1, 2] = height / 2.0
    return intrinsic


def estimate_camera_pose_with_conf(
    pts3d: np.ndarray,
    conf: np.ndarray,
    intrinsic: np.ndarray,
    conf_percentile: float,
    seed: int,
    max_samples: int = 12000,
) -> tuple[np.ndarray, bool, int]:
    height, width = conf.shape
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    valid = np.isfinite(pts3d).all(axis=2) & np.isfinite(conf)
    valid &= conf >= max(1.0, float(np.percentile(conf[np.isfinite(conf)], conf_percentile)))
    valid &= np.linalg.norm(pts3d, axis=2) > 1e-5
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size < 64:
        valid = np.isfinite(pts3d).all(axis=2) & np.isfinite(conf) & (conf > 1.0)
        idx = np.flatnonzero(valid.reshape(-1))
    if idx.size < 64:
        return np.eye(4, dtype=np.float64), False, int(idx.size)

    rng = np.random.default_rng(seed)
    if idx.size > max_samples:
        idx = np.sort(rng.choice(idx, size=max_samples, replace=False))

    object_points = pts3d.reshape(-1, 3)[idx].astype(np.float32)
    image_points = np.column_stack([xs.reshape(-1)[idx], ys.reshape(-1)[idx]]).astype(np.float32)
    dist_coeffs = np.zeros(4, dtype=np.float32)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        intrinsic.astype(np.float32),
        dist_coeffs,
        iterationsCount=120,
        reprojectionError=4.0,
        confidence=0.995,
        flags=cv2.SOLVEPNP_EPNP,
    )
    inlier_count = 0 if inliers is None else int(len(inliers))
    if not success or inlier_count < 48:
        return np.eye(4, dtype=np.float64), False, inlier_count
    try:
        inlier_ids = inliers.reshape(-1)
        success_refined, rvec, tvec = cv2.solvePnP(
            object_points[inlier_ids],
            image_points[inlier_ids],
            intrinsic.astype(np.float32),
            dist_coeffs,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        success = bool(success_refined)
    except cv2.error:
        success = True
    if not success:
        return np.eye(4, dtype=np.float64), False, inlier_count
    rot, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rot
    w2c[:3, 3] = tvec.reshape(3)
    return np.linalg.inv(w2c), True, inlier_count


def calibration_centers(dataset: Path, stream: Stream, scenario: str) -> np.ndarray:
    centers = []
    for frame in stream.frames:
        calib = base.load_calib(dataset, stream.agent, frame, scenario)
        c2w = base.camera_c2w_cv(calib, stream.camera)
        centers.append(c2w[:3, 3])
    return np.asarray(centers, dtype=np.float64)


def load_stream_preds(out: Path, stream: Stream) -> tuple[Path, dict[str, np.ndarray]]:
    result_dir = out / "slam3r_results" / stream.test_name
    preds_dir = result_dir / "preds"
    if not preds_dir.exists():
        raise FileNotFoundError(f"Missing SLAM3R preds for {stream.key}: {preds_dir}")
    arrays = {
        "local_pcds": np.load(preds_dir / "local_pcds.npy"),
        "registered_pcds": np.load(preds_dir / "registered_pcds.npy"),
        "local_confs": np.load(preds_dir / "local_confs.npy"),
        "registered_confs": np.load(preds_dir / "registered_confs.npy"),
        "input_imgs": np.load(preds_dir / "input_imgs.npy"),
    }
    return result_dir, arrays


def points_from_stream(
    stream: Stream,
    arrays: dict[str, np.ndarray],
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    frame_keep_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    pts = arrays["registered_pcds"].astype(np.float32)
    conf = arrays["registered_confs"].astype(np.float32)
    colors = np.clip(arrays["input_imgs"], 0, 255).astype(np.uint8)
    valid = np.isfinite(pts).all(axis=3) & np.isfinite(conf)
    valid &= conf >= max(1.0, float(np.percentile(conf[np.isfinite(conf)], args.point_conf_percentile)))
    valid &= colors.sum(axis=3) >= 16
    if frame_keep_mask is not None:
        valid &= frame_keep_mask[:, None, None]

    candidate_points = pts[valid]
    candidate_colors = colors[valid]
    before_trim = int(len(candidate_points))
    if before_trim == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            {"stream": stream.key, "candidate_points": 0, "exported_points": 0},
        )

    if args.trim_percentile > 0 and len(candidate_points) > 100:
        lo = np.percentile(candidate_points, args.trim_percentile, axis=0)
        hi = np.percentile(candidate_points, 100.0 - args.trim_percentile, axis=0)
        keep = np.all((candidate_points >= lo) & (candidate_points <= hi), axis=1)
        candidate_points = candidate_points[keep]
        candidate_colors = candidate_colors[keep]

    if args.points_per_stream > 0 and len(candidate_points) > args.points_per_stream:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(candidate_points), size=args.points_per_stream, replace=False))
        candidate_points = candidate_points[idx]
        candidate_colors = candidate_colors[idx]

    world_points = apply_similarity(candidate_points, scale, rot, trans)
    in_world_bounds = np.isfinite(world_points).all(axis=1)
    if args.min_world_z is not None:
        in_world_bounds &= world_points[:, 2] >= float(args.min_world_z)
    if args.max_world_z is not None:
        in_world_bounds &= world_points[:, 2] <= float(args.max_world_z)
    removed_by_world_bounds = int(len(world_points) - np.count_nonzero(in_world_bounds))
    world_points = world_points[in_world_bounds]
    candidate_colors = candidate_colors[in_world_bounds]
    summary = {
        "stream": stream.key,
        "candidate_points": before_trim,
        "exported_points": int(len(world_points)),
        "confidence_percentile": float(args.point_conf_percentile),
        "accepted_frame_count": int(frame_keep_mask.sum()) if frame_keep_mask is not None else len(stream.frames),
        "removed_by_world_bounds": removed_by_world_bounds,
        "world_z_filter": [args.min_world_z, args.max_world_z],
    }
    if len(world_points):
        summary["bbox_min_world"] = [float(x) for x in world_points.min(axis=0)]
        summary["bbox_max_world"] = [float(x) for x in world_points.max(axis=0)]
    return world_points, candidate_colors, summary


def sky_like_colors(colors: np.ndarray, ys: np.ndarray, height: int) -> np.ndarray:
    rgb = colors.astype(np.int16)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    high = ys < height * 0.58
    blue_sky = (b > 115) & (b > r + 18) & (b > g + 8)
    pale_sky = (b > 145) & (g > 135) & (r > 115) & (np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b]) < 45)
    return high & (blue_sky | pale_sky)


def load_static_mask_flat(mask_export: Path, stream: Stream, frame: int, height: int, width: int) -> np.ndarray | None:
    mask_path = mask_export / "masks" / stream.agent / stream.camera / f"frame_{frame:03d}.png"
    if not mask_path.exists():
        return None
    mask = Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(mask) <= 0).reshape(-1)


def points_from_stream_camera_ray_depth(
    stream: Stream,
    arrays: dict[str, np.ndarray],
    slam_c2w_poses: list[np.ndarray | None],
    scale: float,
    args: argparse.Namespace,
    seed: int,
    frame_keep_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    pts = arrays["registered_pcds"].astype(np.float32)
    conf = arrays["registered_confs"].astype(np.float32)
    colors = np.clip(arrays["input_imgs"], 0, 255).astype(np.uint8)
    height, width = conf.shape[1:3]
    ys = np.repeat(np.arange(height), width)

    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    candidate_points = 0
    exported_before_sample = 0
    removed_by_mask = 0
    skipped_without_mask = 0
    used_mask_frames = 0
    used_frames = 0

    for frame_idx, frame in enumerate(stream.frames):
        if not frame_keep_mask[frame_idx]:
            continue
        slam_c2w = slam_c2w_poses[frame_idx]
        if slam_c2w is None:
            continue

        frame_points = pts[frame_idx].reshape(-1, 3).astype(np.float64)
        frame_conf = conf[frame_idx].reshape(-1)
        frame_colors = colors[frame_idx].reshape(-1, 3)
        valid = np.isfinite(frame_points).all(axis=1) & np.isfinite(frame_conf)
        valid &= frame_conf >= max(1.0, float(np.percentile(frame_conf[np.isfinite(frame_conf)], args.point_conf_percentile)))
        valid &= frame_colors.sum(axis=1) >= 16
        candidate_points += int(np.count_nonzero(valid))

        if not args.disable_sky_filter:
            valid &= ~sky_like_colors(frame_colors, ys, height)

        if not args.disable_mask_filter:
            static_mask = load_static_mask_flat(args.mask_export, stream, frame, height, width)
            if static_mask is None:
                if args.require_mask_filter:
                    skipped_without_mask += 1
                    continue
            else:
                used_mask_frames += 1
                before = int(np.count_nonzero(valid))
                valid &= static_mask
                removed_by_mask += before - int(np.count_nonzero(valid))

        if not np.any(valid):
            continue

        slam_w2c = np.linalg.inv(slam_c2w)
        cam_points = (slam_w2c[:3, :3] @ frame_points.T).T + slam_w2c[:3, 3]
        depth = cam_points[:, 2]
        valid &= depth >= args.min_camera_depth
        valid &= depth <= args.max_camera_depth
        valid &= depth * scale <= args.max_metric_depth_m

        if args.depth_trim_percentile > 0 and np.count_nonzero(valid) > 100:
            kept_depth = depth[valid]
            lo = np.percentile(kept_depth, args.depth_trim_percentile)
            hi = np.percentile(kept_depth, 100.0 - args.depth_trim_percentile)
            valid &= (depth >= lo) & (depth <= hi)

        if not np.any(valid):
            continue

        calib = base.load_calib(args.dataset, stream.agent, frame, args.scenario)
        calib_c2w = base.camera_c2w_cv(calib, stream.camera)
        metric_camera_points = cam_points[valid] * scale
        world_points = (calib_c2w[:3, :3] @ metric_camera_points.T).T + calib_c2w[:3, 3]
        in_world_bounds = np.isfinite(world_points).all(axis=1)
        if args.min_world_z is not None:
            in_world_bounds &= world_points[:, 2] >= float(args.min_world_z)
        if args.max_world_z is not None:
            in_world_bounds &= world_points[:, 2] <= float(args.max_world_z)
        if not np.any(in_world_bounds):
            continue

        point_parts.append(world_points[in_world_bounds].astype(np.float32))
        color_parts.append(frame_colors[valid][in_world_bounds])
        exported_before_sample += int(np.count_nonzero(in_world_bounds))
        used_frames += 1

    if not point_parts:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            {
                "stream": stream.key,
                "candidate_points": candidate_points,
                "exported_points": 0,
                "fusion_mode": args.fusion_mode,
                "accepted_frame_count": 0,
                "skipped_without_mask": skipped_without_mask,
                "used_mask_frames": used_mask_frames,
                "removed_by_mask": removed_by_mask,
                "world_z_filter": [args.min_world_z, args.max_world_z],
            },
        )

    world_points = np.vstack(point_parts)
    point_colors = np.vstack(color_parts)
    if args.points_per_stream > 0 and len(world_points) > args.points_per_stream:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(world_points), size=args.points_per_stream, replace=False))
        world_points = world_points[idx]
        point_colors = point_colors[idx]

    summary = {
        "stream": stream.key,
        "candidate_points": candidate_points,
        "exported_before_sample": exported_before_sample,
        "exported_points": int(len(world_points)),
        "fusion_mode": args.fusion_mode,
        "confidence_percentile": float(args.point_conf_percentile),
        "accepted_frame_count": used_frames,
        "frame_alignment_accepted_count": int(frame_keep_mask.sum()),
        "skipped_without_mask": skipped_without_mask,
        "used_mask_frames": used_mask_frames,
        "removed_by_mask": removed_by_mask,
        "camera_depth_filter": [args.min_camera_depth, args.max_camera_depth],
        "max_metric_depth_m": args.max_metric_depth_m,
        "world_z_filter": [args.min_world_z, args.max_world_z],
    }
    if len(world_points):
        summary["bbox_min_world"] = [float(x) for x in world_points.min(axis=0)]
        summary["bbox_max_world"] = [float(x) for x in world_points.max(axis=0)]
    return world_points, point_colors, summary


def fuse_streams(streams: list[Stream], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    dataset = args.dataset.resolve()
    slam3r_root = args.slam3r_root.resolve()
    points_parts = []
    colors_parts = []
    stream_alignments = []
    stream_summaries = []

    for stream_index, stream in enumerate(streams):
        _result_dir, arrays = load_stream_preds(args.out, stream)
        intrinsic = estimate_intrinsic(arrays["local_pcds"], slam3r_root)
        local_centers = []
        target_centers = calibration_centers(dataset, stream, args.scenario)
        pose_records = []
        slam_c2w_poses: list[np.ndarray | None] = []
        for frame_idx in range(len(stream.frames)):
            c2w, ok, inliers = estimate_camera_pose_with_conf(
                arrays["registered_pcds"][frame_idx],
                arrays["registered_confs"][frame_idx],
                intrinsic,
                args.pose_conf_percentile,
                seed=args.seed + stream_index * 1000 + frame_idx,
            )
            if ok:
                local_centers.append(c2w[:3, 3])
                slam_c2w_poses.append(c2w)
            else:
                slam_c2w_poses.append(None)
            pose_records.append(
                {
                    "frame": int(stream.frames[frame_idx]),
                    "ok": bool(ok),
                    "pnp_inliers": int(inliers),
                    "local_camera_center": c2w[:3, 3].tolist() if ok else None,
                    "target_camera_center_world": target_centers[frame_idx].tolist(),
                }
            )
        ok_mask = np.asarray([record["ok"] for record in pose_records], dtype=bool)
        local_centers_arr = np.asarray(local_centers, dtype=np.float64)
        target_centers_arr = target_centers[ok_mask]
        if len(local_centers_arr) < 3:
            raise RuntimeError(f"Not enough camera poses estimated for {stream.key}: {len(local_centers_arr)}")
        scale, rot, trans, residual = fit_similarity_umeyama(local_centers_arr, target_centers_arr)
        ok_indices = np.flatnonzero(ok_mask)
        local_aligned = (scale * (rot @ local_centers_arr.T)).T + trans
        pose_errors = np.linalg.norm(local_aligned - target_centers_arr, axis=1)
        frame_keep_mask = np.zeros(len(stream.frames), dtype=bool)
        for local_idx, frame_idx in enumerate(ok_indices):
            error_m = float(pose_errors[local_idx])
            pose_records[int(frame_idx)]["aligned_pose_error_m"] = error_m
            if error_m <= args.max_frame_alignment_error_m:
                frame_keep_mask[int(frame_idx)] = True

        rejection_reasons = []
        if residual["rmse_m"] > args.max_stream_rmse_m:
            rejection_reasons.append(f"rmse>{args.max_stream_rmse_m}")
        if residual["median_error_m"] > args.max_stream_median_error_m:
            rejection_reasons.append(f"median>{args.max_stream_median_error_m}")
        if residual["max_error_m"] > args.max_stream_max_error_m:
            rejection_reasons.append(f"max>{args.max_stream_max_error_m}")
        if int(frame_keep_mask.sum()) < 3:
            rejection_reasons.append("accepted_frames<3")
        accepted_for_fusion = not rejection_reasons

        if accepted_for_fusion:
            if args.fusion_mode == "camera-ray-depth":
                points, colors, point_summary = points_from_stream_camera_ray_depth(
                    stream,
                    arrays,
                    slam_c2w_poses,
                    scale,
                    args,
                    seed=args.seed + stream_index,
                    frame_keep_mask=frame_keep_mask,
                )
            else:
                points, colors, point_summary = points_from_stream(
                    stream,
                    arrays,
                    scale,
                    rot,
                    trans,
                    args,
                    seed=args.seed + stream_index,
                    frame_keep_mask=frame_keep_mask,
                )
            points_parts.append(points)
            colors_parts.append(colors)
        else:
            point_summary = {
                "stream": stream.key,
                "candidate_points": 0,
                "exported_points": 0,
                "fusion_mode": args.fusion_mode,
                "confidence_percentile": float(args.point_conf_percentile),
                "accepted_frame_count": int(frame_keep_mask.sum()),
                "removed_by_world_bounds": 0,
                "world_z_filter": [args.min_world_z, args.max_world_z],
            }
        point_summary["accepted_for_fusion"] = accepted_for_fusion
        point_summary["rejection_reasons"] = rejection_reasons
        stream_summaries.append(point_summary)
        stream_alignments.append(
            {
                "stream": stream.key,
                "agent": stream.agent,
                "camera": stream.camera,
                "frame_count": len(stream.frames),
                "estimated_pose_count": int(len(local_centers_arr)),
                "intrinsic_224": intrinsic.tolist(),
                "method": "umeyama_similarity_fit_slam3r_pnp_camera_centers_to_deepaccident_calibration",
                "point_fusion_method": args.fusion_mode,
                "scale": float(scale),
                "rotation": rot.tolist(),
                "translation": trans.tolist(),
                "residual": residual,
                "accepted_for_fusion": accepted_for_fusion,
                "accepted_frame_count": int(frame_keep_mask.sum()),
                "rejection_reasons": rejection_reasons,
                "poses": pose_records,
            }
        )

    points_world = np.vstack(points_parts) if points_parts else np.zeros((0, 3), dtype=np.float32)
    colors = np.vstack(colors_parts) if colors_parts else np.zeros((0, 3), dtype=np.uint8)
    if args.max_points > 0 and len(points_world) > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(points_world), size=args.max_points, replace=False))
        points_world = points_world[idx]
        colors = colors[idx]

    residual_errors = [
        error
        for alignment in stream_alignments
        if alignment.get("accepted_for_fusion")
        for error in alignment["residual"].get("errors_m", [])
    ]
    if residual_errors:
        arr = np.asarray(residual_errors, dtype=np.float64)
        aggregate_residual = {
            "camera_count": int(len(arr)),
            "rmse_m": float(np.sqrt(np.mean(arr * arr))),
            "median_error_m": float(np.median(arr)),
            "max_error_m": float(np.max(arr)),
        }
    else:
        aggregate_residual = {"camera_count": 0, "rmse_m": None, "median_error_m": None, "max_error_m": None}
    alignment = {
        "method": "streamwise_slam3r_to_deepaccident_world_alignment",
        "aggregate_residual": aggregate_residual,
        "accepted_stream_count": int(sum(1 for item in stream_alignments if item.get("accepted_for_fusion"))),
        "rejected_stream_count": int(sum(1 for item in stream_alignments if not item.get("accepted_for_fusion"))),
        "filters": {
            "max_stream_rmse_m": args.max_stream_rmse_m,
            "max_stream_median_error_m": args.max_stream_median_error_m,
            "max_stream_max_error_m": args.max_stream_max_error_m,
            "max_frame_alignment_error_m": args.max_frame_alignment_error_m,
            "world_z": [args.min_world_z, args.max_world_z],
        },
        "streams": stream_alignments,
        "point_summaries": stream_summaries,
    }
    return points_world, colors, alignment


def voxel_downsample_world(points: np.ndarray, colors: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    if voxel <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return points[unique_idx], colors[unique_idx]


def write_reports(out: Path, args: argparse.Namespace, summary: dict, alignment: dict, diagnostics: dict) -> None:
    quality = f"""# SLAM3R DeepAccident Reconstruction Quality Report

Status: `{summary['status']}`

## Inputs

- Dataset: `{args.dataset.resolve()}`
- Scenario: `{args.scenario}`
- Agents: `{', '.join(args.agents)}`
- Cameras: `{', '.join(args.cameras)}`
- RGB frames exported: `{summary['input_frame_count']}`
- SLAM3R commit: `{summary['slam3r_commit']}`

## Reconstruction

- Backend: `{summary['backend']}`
- Fusion mode: `{args.fusion_mode}`
- Stream count: `{summary['stream_count']}`
- Accepted streams: `{summary['accepted_stream_count']}`
- Rejected streams: `{summary['rejected_stream_count']}`
- Exported world points before voxel: `{summary['points_before_voxel']}`
- Exported points after voxel/sample: `{summary['exported_point_count']}`
- Alignment camera centers: `{alignment['aggregate_residual']['camera_count']}`
- Alignment RMSE m: `{alignment['aggregate_residual']['rmse_m']}`
- Alignment median error m: `{alignment['aggregate_residual']['median_error_m']}`
- LiDAR geometry used: `false`
- Stream filters: RMSE <= `{alignment['filters']['max_stream_rmse_m']}` m, median <= `{alignment['filters']['max_stream_median_error_m']}` m, max <= `{alignment['filters']['max_stream_max_error_m']}` m.
- Frame/world filters: frame alignment error <= `{alignment['filters']['max_frame_alignment_error_m']}` m, world z in `{alignment['filters']['world_z']}`.

## Caveats

- SLAM3R is monocular and is run per camera stream; streams are fused by DeepAccident calibration camera centers.
- SLAM3R crops frames to 224x224, so wide-angle border content is not reconstructed.
- Dynamic traffic is filtered when matching masks exist under `{args.mask_export}`; unmasked frames may still retain transient geometry.
- In `camera-ray-depth` mode, SLAM3R provides camera-frame depth while DeepAccident calibration provides the final camera pose.

## Outputs

- `reconstruction/points_world.ply`
- `reconstruction/points.ply`
- `scene.glb`
- `reports/preview_point_cloud.png`
- `reconstruction/alignment.json`
- `replay/agent_tracks.json`
- `viewer/index.html`
"""
    (out / "reports" / "quality_report.md").write_text(quality, encoding="utf-8")

    no_lidar = """# No-LiDAR Audit

Status: pass for `scripts/run_slam3r_deepaccident_reconstruction.py`.

Inputs used:

- DeepAccident RGB camera frames.
- DeepAccident calibration files for camera-center world alignment and vehicle tracks.

Inputs not used:

- `lidar01` point clouds: not read.
- Historical LiDAR-derived PLY/GLB outputs: not read.
- Proxy meshes or simulator map meshes: not read.
"""
    (out / "reports" / "no_lidar_audit.md").write_text(no_lidar, encoding="utf-8")

    diagnostics_report = {
        "closest_approach": diagnostics.get("closest_approach"),
        "sequence_summary": diagnostics.get("sequence_summary"),
    }
    base.write_json(out / "reports" / "accident_context_summary.json", diagnostics_report)


def git_commit(path: Path) -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, text=True, capture_output=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.dataset = args.dataset.resolve()
    args.out = args.out.resolve()
    args.slam3r_root = args.slam3r_root.resolve()
    args.mask_export = args.mask_export.resolve()
    base.CATEGORY = args.category
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.slam3r_root.exists():
        raise FileNotFoundError(f"SLAM3R root does not exist: {args.slam3r_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("SLAM3R reconstruction requires CUDA.")

    start = time.time()
    clean_output(args.out, keep_slam3r=args.skip_slam3r or args.resume_slam3r)
    streams = export_stream_inputs(args.dataset, args.out, args)
    stream_runs = run_slam3r_streams(streams, args, repo)
    base.write_json(args.out / "reconstruction" / "stream_runs.json", {"streams": stream_runs})

    points_world, colors, alignment = fuse_streams(streams, args)
    points_before_voxel = int(len(points_world))
    points_world, colors = voxel_downsample_world(points_world, colors, args.voxel)
    if args.max_points > 0 and len(points_world) > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(points_world), size=args.max_points, replace=False))
        points_world = points_world[idx]
        colors = colors[idx]
    if len(points_world) == 0:
        raise RuntimeError("SLAM3R fusion produced no points.")

    viewer_points, origin = base.world_to_viewer(points_world)
    base.write_ply(args.out / "reconstruction" / "points_world.ply", points_world, colors)
    base.write_ply(args.out / "reconstruction" / "points.ply", viewer_points, colors)
    base.export_scene_glb(args.out / "scene.glb", viewer_points, colors)
    base.write_preview(args.out / "reports" / "preview_point_cloud.png", viewer_points, colors)

    alignment["viewer_origin_world"] = origin.tolist()
    base.write_json(args.out / "reconstruction" / "alignment.json", alignment)

    all_frames = sorted({frame for stream in streams for frame in stream.frames})
    tracks = multi.build_agent_tracks(args.dataset, args.out, args.agents, all_frames, args.scenario, origin, args.nominal_fps)
    object_context = multi.build_object_tracks(args.dataset, args.out, args.agents, all_frames, args.scenario)
    diagnostics = multi.accident_diagnostics(args.out, tracks)
    multi.write_viewer(args.out)

    camera_records = []
    for stream in streams:
        for frame in stream.frames:
            calib = base.load_calib(args.dataset, stream.agent, frame, args.scenario)
            camera_records.append(
                {
                    "agent": stream.agent,
                    "camera": stream.camera,
                    "frame": int(frame),
                    "camera_to_world_cv": base.camera_c2w_cv(calib, stream.camera).tolist(),
                }
            )
    base.write_json(
        args.out / "reconstruction" / "cameras.json",
        {
            "coordinate_note": "DeepAccident world frame; viewer maps world x,z,y after origin subtraction.",
            "world_origin_carla": origin.tolist(),
            "views": camera_records,
        },
    )

    elapsed = time.time() - start
    status = "ok" if len(points_world) >= 100000 else "weak"
    backend = (
        "slam3r_camera_ray_depth_calibrated"
        if args.fusion_mode == "camera-ray-depth"
        else "slam3r_streamwise_world_aligned"
    )
    summary = {
        "schema_version": "0.1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "backend": backend,
        "fusion_mode": args.fusion_mode,
        "slam3r_commit": git_commit(args.slam3r_root),
        "lidar_used": False,
        "legacy_lidar_assets_used": False,
        "dataset_root": str(args.dataset),
        "category": args.category,
        "scenario": args.scenario,
        "stream_count": len(streams),
        "accepted_stream_count": int(alignment["accepted_stream_count"]),
        "rejected_stream_count": int(alignment["rejected_stream_count"]),
        "input_frame_count": int(sum(len(stream.frames) for stream in streams)),
        "points_before_voxel": points_before_voxel,
        "exported_point_count": int(len(points_world)),
        "alignment_residual": alignment["aggregate_residual"],
        "object_context_records": int(object_context["record_count"]),
        "elapsed_seconds": elapsed,
        "outputs": {
            "points_ply": "reconstruction/points.ply",
            "points_world_ply": "reconstruction/points_world.ply",
            "scene_glb": "scene.glb",
            "preview": "reports/preview_point_cloud.png",
            "quality_report": "reports/quality_report.md",
            "alignment": "reconstruction/alignment.json",
            "stream_runs": "reconstruction/stream_runs.json",
            "viewer": "viewer/index.html",
            "agent_tracks": "replay/agent_tracks.json",
            "accident_diagnostics": "replay/accident_diagnostics.json",
        },
    }
    if len(points_world):
        summary["bbox_min_world"] = [float(x) for x in points_world.min(axis=0)]
        summary["bbox_max_world"] = [float(x) for x in points_world.max(axis=0)]

    base.write_json(args.out / "summary.json", summary)
    manifest = {
        "schema_version": "0.1",
        "case_id": "town04_type1_subtype2_slam3r_reconstruction",
        "status": status,
        "created_utc": summary["created_utc"],
        "inputs": {
            "dataset_root": str(args.dataset),
            "category": args.category,
            "scenario": args.scenario,
            "agents": args.agents,
            "cameras": args.cameras,
            "rgb_frame_count": summary["input_frame_count"],
            "slam3r_root": str(args.slam3r_root),
            "slam3r_commit": summary["slam3r_commit"],
        },
        "outputs": summary["outputs"],
        "quality_summary": {
            "backend": summary["backend"],
            "point_count": summary["exported_point_count"],
            "accepted_stream_count": summary["accepted_stream_count"],
            "rejected_stream_count": summary["rejected_stream_count"],
            "alignment_rmse_m": summary["alignment_residual"]["rmse_m"],
            "alignment_median_error_m": summary["alignment_residual"]["median_error_m"],
            "object_context_records": summary["object_context_records"],
        },
        "lidar_used": False,
        "legacy_lidar_assets_used": False,
    }
    base.write_json(args.out / "manifest.json", manifest)
    write_reports(args.out, args, summary, alignment, diagnostics)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
