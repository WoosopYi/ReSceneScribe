#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement


SCENARIO = "Town03_type001_subtype0001_scenario00024"
DEFAULT_CATEGORY = "type1_subtype1_accident"
CATEGORY = DEFAULT_CATEGORY
AGENTS = ["ego_vehicle", "ego_vehicle_behind", "other_vehicle", "other_vehicle_behind"]
CAMERAS = ["Camera_Front"]
DYNAMIC_CLASS_NAMES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}

# DeepAccident camera coordinates use X forward, Y right, Z up. OpenCV/COLMAP use
# X right, Y down, Z forward.
CARLA_CAM_TO_CV_CAM = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class View:
    view_id: str
    agent: str
    camera: str
    frame: int
    src_path: Path
    rgb_rel: str
    mask_rel: str
    masked_rel: str
    K: np.ndarray
    c2w: np.ndarray
    w2c: np.ndarray
    P: np.ndarray


def dataset_path(root: Path, agent: str, sensor: str, scenario: str) -> Path:
    return root / CATEGORY / agent / sensor / scenario


def load_calib(root: Path, agent: str, frame: int, scenario: str) -> dict:
    path = dataset_path(root, agent, "calib", scenario) / f"{scenario}_{frame:03d}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def camera_c2w_cv(calib: dict, camera: str) -> np.ndarray:
    lidar_to_cam = np.asarray(calib[f"lidar_to_{camera}"], dtype=np.float64)
    c2w_carla_camera = (
        np.asarray(calib["ego_to_world"], dtype=np.float64)
        @ np.asarray(calib["lidar_to_ego"], dtype=np.float64)
        @ np.linalg.inv(lidar_to_cam)
    )
    cv_to_carla = np.eye(4, dtype=np.float64)
    cv_to_carla[:3, :3] = CARLA_CAM_TO_CV_CAM.T
    return c2w_carla_camera @ cv_to_carla


def ensure_clean_output(out: Path) -> None:
    for rel in [
        "rgb_frames",
        "masks",
        "masked_frames",
        "reconstruction",
        "calibration",
        "replay",
        "viewer",
        "reports",
        "logs",
    ]:
        target = out / rel
        if target.exists():
            shutil.rmtree(target)
    for rel in ["manifest.json", "mask_manifest.json", "scene.glb"]:
        target = out / rel
        if target.exists():
            target.unlink()
    for rel in [
        "rgb_frames",
        "masks",
        "masked_frames",
        "reconstruction",
        "reconstruction/depth",
        "calibration",
        "replay",
        "viewer",
        "reports",
        "logs",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)


def select_frames(start: int, end: int, step: int, include_end: bool = True) -> list[int]:
    frames = list(range(start, end + 1, step))
    if include_end and end not in frames:
        frames.append(end)
    return sorted(set(frames))


def copy_inputs_and_build_views(
    dataset: Path,
    out: Path,
    agents: list[str],
    cameras: list[str],
    frames: list[int],
    scenario: str,
) -> list[View]:
    views: list[View] = []
    for agent in agents:
        for camera in cameras:
            for frame in frames:
                src = dataset_path(dataset, agent, camera, scenario) / f"{scenario}_{frame:03d}.jpg"
                if not src.exists():
                    continue
                rgb_rel = f"rgb_frames/{agent}/{camera}/frame_{frame:03d}.jpg"
                mask_rel = f"masks/{agent}/{camera}/frame_{frame:03d}.png"
                masked_rel = f"masked_frames/{agent}/{camera}/frame_{frame:03d}.jpg"
                dst = out / rgb_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

                calib = load_calib(dataset, agent, frame, scenario)
                K = np.asarray(calib[f"intrinsic_{camera}"], dtype=np.float64)
                c2w = camera_c2w_cv(calib, camera)
                w2c = np.linalg.inv(c2w)
                P = K @ w2c[:3, :]
                views.append(
                    View(
                        view_id=f"{agent}:{camera}:{frame:03d}",
                        agent=agent,
                        camera=camera,
                        frame=frame,
                        src_path=src,
                        rgb_rel=rgb_rel,
                        mask_rel=mask_rel,
                        masked_rel=masked_rel,
                        K=K,
                        c2w=c2w,
                        w2c=w2c,
                        P=P,
                    )
                )
    return views


def run_yolo_masks(out: Path, views: list[View], model_name: str, imgsz: int) -> dict:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "backend": "ultralytics",
            "status": "unavailable",
            "error": repr(exc),
            "masked_pixels": 0,
        }

    try:
        model = YOLO(model_name)
    except Exception as exc:  # pragma: no cover - network/model dependent
        return {
            "backend": "ultralytics",
            "status": "model_load_failed",
            "model": model_name,
            "error": repr(exc),
            "masked_pixels": 0,
        }

    class_names = model.names
    total_masked = 0
    frames = []
    for view in views:
        image_path = out / view.rgb_rel
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        result = model.predict(str(image_path), imgsz=imgsz, verbose=False)[0]

        if result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None else []
            for i, cls_idx in enumerate(classes):
                if str(class_names.get(int(cls_idx), "")).lower() in DYNAMIC_CLASS_NAMES:
                    resized = cv2.resize(masks[i], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask[resized > 0.5] = 255
        elif result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, cls_idx in zip(boxes, classes):
                if str(class_names.get(int(cls_idx), "")).lower() not in DYNAMIC_CLASS_NAMES:
                    continue
                x1, y1, x2, y2 = np.round(box).astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)
                mask[y1:y2 + 1, x1:x2 + 1] = 255

        masked = image.copy()
        masked[mask > 0] = 0
        mask_path = out / view.mask_rel
        masked_path = out / view.masked_rel
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        masked_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)
        Image.fromarray(masked).save(masked_path, quality=92)
        masked_pixels = int(np.count_nonzero(mask))
        total_masked += masked_pixels
        frames.append(
            {
                "view_id": view.view_id,
                "agent": view.agent,
                "camera": view.camera,
                "frame_index": view.frame,
                "rgb_path": view.rgb_rel,
                "mask_path": view.mask_rel,
                "masked_frame_path": view.masked_rel,
                "mask_status": "detector_mask" if masked_pixels else "detector_empty",
                "qa_status": "algorithmic_detector_unreviewed",
                "masked_pixels": masked_pixels,
            }
        )

    return {
        "backend": "ultralytics",
        "status": "ok",
        "model": model_name,
        "masked_pixels": total_masked,
        "frames": frames,
    }


def write_empty_masks(out: Path, views: list[View]) -> dict:
    frames = []
    for view in views:
        image = np.asarray(Image.open(out / view.rgb_rel).convert("RGB"))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        Image.fromarray(mask).save(out / view.mask_rel)
        Image.fromarray(image).save(out / view.masked_rel, quality=92)
        frames.append(
            {
                "view_id": view.view_id,
                "agent": view.agent,
                "camera": view.camera,
                "frame_index": view.frame,
                "rgb_path": view.rgb_rel,
                "mask_path": view.mask_rel,
                "masked_frame_path": view.masked_rel,
                "mask_status": "empty_fallback",
                "qa_status": "not_completion_grade",
                "masked_pixels": 0,
            }
        )
    return {
        "backend": "empty",
        "status": "fallback_only",
        "masked_pixels": 0,
        "frames": frames,
    }


def make_detector() -> tuple[object, str]:
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=5000), "SIFT"
    return cv2.ORB_create(nfeatures=7000), "ORB"


def load_features(out: Path, view: View, detector: object) -> tuple[list[cv2.KeyPoint], np.ndarray | None, np.ndarray]:
    img = cv2.imread(str(out / view.masked_rel), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(out / view.mask_rel), cv2.IMREAD_GRAYSCALE)
    static_mask = None
    if mask is not None:
        static_mask = np.where(mask > 0, 0, 255).astype(np.uint8)
    keypoints, desc = detector.detectAndCompute(img, static_mask)
    rgb = cv2.cvtColor(cv2.imread(str(out / view.rgb_rel), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return keypoints, desc, rgb


def match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray, detector_name: str) -> list[cv2.DMatch]:
    if desc_a is None or desc_b is None or len(desc_a) < 8 or len(desc_b) < 8:
        return []
    norm = cv2.NORM_L2 if detector_name == "SIFT" else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm)
    raw = matcher.knnMatch(desc_a, desc_b, k=2)
    good: list[cv2.DMatch] = []
    for pair in raw:
        if len(pair) != 2:
            continue
        a, b = pair
        if a.distance < 0.75 * b.distance:
            good.append(a)
    return good


def reprojection_error(P: np.ndarray, X: np.ndarray, pts: np.ndarray) -> np.ndarray:
    hom = np.column_stack([X, np.ones(len(X), dtype=np.float64)])
    proj = (P @ hom.T).T
    xy = proj[:, :2] / np.maximum(np.abs(proj[:, 2:3]), 1e-9)
    return np.linalg.norm(xy - pts, axis=1)


def depth_in_camera(w2c: np.ndarray, X: np.ndarray) -> np.ndarray:
    hom = np.column_stack([X, np.ones(len(X), dtype=np.float64)])
    cam = (w2c @ hom.T).T
    return cam[:, 2]


def triangulation_angles(c1: np.ndarray, c2: np.ndarray, X: np.ndarray) -> np.ndarray:
    r1 = X - c1[None, :]
    r2 = X - c2[None, :]
    r1 /= np.maximum(np.linalg.norm(r1, axis=1, keepdims=True), 1e-9)
    r2 /= np.maximum(np.linalg.norm(r2, axis=1, keepdims=True), 1e-9)
    cosang = np.clip(np.sum(r1 * r2, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def candidate_pairs(views: list[View], max_pairs: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    by_stream: dict[tuple[str, str], list[int]] = {}
    for i, view in enumerate(views):
        by_stream.setdefault((view.agent, view.camera), []).append(i)
    for _, idxs in by_stream.items():
        idxs = sorted(idxs, key=lambda i: views[i].frame)
        for offset in [1, 2, 3]:
            for a, b in zip(idxs, idxs[offset:]):
                pairs.append((a, b))
    by_frame_agent: dict[tuple[str, int], list[int]] = {}
    for i, view in enumerate(views):
        by_frame_agent.setdefault((view.agent, view.frame), []).append(i)
    for _, idxs in by_frame_agent.items():
        idxs = sorted(idxs, key=lambda i: views[i].camera)
        for i, a in enumerate(idxs):
            for b in idxs[i + 1 :]:
                pairs.append((a, b))
    filtered = []
    seen: set[tuple[int, int]] = set()
    for a, b in pairs:
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        c1 = views[a].c2w[:3, 3]
        c2 = views[b].c2w[:3, 3]
        baseline = float(np.linalg.norm(c1 - c2))
        if baseline > 0.15:
            filtered.append((a, b))
    # Preserve construction order: adjacent same-stream pairs first, then wider
    # temporal offsets, then same-time multi-camera pairs. Dashcam views often
    # lose overlap quickly, so largest-baseline-first selection produced too many
    # empty triangulations.
    return filtered[:max_pairs]


def triangulate_rgb_points(
    out: Path,
    views: list[View],
    max_pairs: int,
    max_reproj_error: float,
    min_angle_deg: float,
    max_depth_m: float,
) -> dict:
    detector, detector_name = make_detector()
    features = [load_features(out, view, detector) for view in views]
    points_all: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []
    pair_reports = []

    for ia, ib in candidate_pairs(views, max_pairs):
        view_a = views[ia]
        view_b = views[ib]
        kpa, desca, rgba = features[ia]
        kpb, descb, _rgbb = features[ib]
        matches = match_descriptors(desca, descb, detector_name)
        if len(matches) < 16:
            pair_reports.append({"pair": [view_a.view_id, view_b.view_id], "status": "few_matches", "matches": len(matches)})
            continue
        pts_a = np.float32([kpa[m.queryIdx].pt for m in matches])
        pts_b = np.float32([kpb[m.trainIdx].pt for m in matches])
        F, inlier_mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 2.0, 0.99)
        if inlier_mask is None:
            pair_reports.append({"pair": [view_a.view_id, view_b.view_id], "status": "ransac_failed", "matches": len(matches)})
            continue
        inliers = inlier_mask.ravel().astype(bool)
        pts_a = pts_a[inliers]
        pts_b = pts_b[inliers]
        if len(pts_a) < 12:
            pair_reports.append({"pair": [view_a.view_id, view_b.view_id], "status": "few_inliers", "matches": len(matches), "inliers": int(len(pts_a))})
            continue

        X_h = cv2.triangulatePoints(view_a.P, view_b.P, pts_a.T, pts_b.T).T
        X = X_h[:, :3] / np.maximum(np.abs(X_h[:, 3:4]), 1e-9)
        z1 = depth_in_camera(view_a.w2c, X)
        z2 = depth_in_camera(view_b.w2c, X)
        e1 = reprojection_error(view_a.P, X, pts_a)
        e2 = reprojection_error(view_b.P, X, pts_b)
        angles = triangulation_angles(view_a.c2w[:3, 3], view_b.c2w[:3, 3], X)
        dist = np.linalg.norm(X - view_a.c2w[:3, 3][None, :], axis=1)
        keep = (
            np.isfinite(X).all(axis=1)
            & (z1 > 0.5)
            & (z2 > 0.5)
            & (dist < max_depth_m)
            & (e1 < max_reproj_error)
            & (e2 < max_reproj_error)
            & (angles > min_angle_deg)
            & (angles < 160.0)
        )
        X = X[keep]
        pts_color = pts_a[keep]
        if len(X) == 0:
            pair_reports.append({
                "pair": [view_a.view_id, view_b.view_id],
                "status": "filtered_empty",
                "matches": len(matches),
                "inliers": int(len(pts_a)),
            })
            continue
        px = np.clip(np.round(pts_color[:, 0]).astype(int), 0, rgba.shape[1] - 1)
        py = np.clip(np.round(pts_color[:, 1]).astype(int), 0, rgba.shape[0] - 1)
        colors = rgba[py, px]
        points_all.append(X.astype(np.float64))
        colors_all.append(colors.astype(np.uint8))
        pair_reports.append({
            "pair": [view_a.view_id, view_b.view_id],
            "status": "ok",
            "matches": len(matches),
            "inliers": int(len(pts_a)),
            "kept_points": int(len(X)),
            "median_reprojection_error_px": float(np.median(np.concatenate([e1[keep], e2[keep]]))),
            "median_triangulation_angle_deg": float(np.median(angles[keep])),
        })

    if not points_all:
        return {
            "backend": f"opencv_{detector_name.lower()}_known_pose_triangulation",
            "status": "failed",
            "points_world": np.zeros((0, 3), dtype=np.float64),
            "colors": np.zeros((0, 3), dtype=np.uint8),
            "pair_reports": pair_reports,
        }

    points = np.vstack(points_all)
    colors = np.vstack(colors_all)
    return {
        "backend": f"opencv_{detector_name.lower()}_known_pose_triangulation",
        "status": "ok",
        "points_world": points,
        "colors": colors,
        "pair_reports": pair_reports,
    }


def world_to_viewer(points: np.ndarray, origin: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if origin is None:
        origin = np.median(points, axis=0) if len(points) else np.zeros(3, dtype=np.float64)
    centered = points - origin[None, :]
    viewer = np.column_stack([centered[:, 0], centered[:, 2], centered[:, 1]])
    return viewer, origin


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel: float, limit: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    if len(unique_idx) > limit:
        rng = np.random.default_rng(42)
        unique_idx = np.sort(rng.choice(unique_idx, size=limit, replace=False))
    return points[unique_idx], colors[unique_idx]


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertex = np.empty(
        len(points),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"] = points[:, 0].astype(np.float32)
    vertex["y"] = points[:, 1].astype(np.float32)
    vertex["z"] = points[:, 2].astype(np.float32)
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def export_scene_glb(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    rgba = np.column_stack([colors, np.full(len(colors), 255, dtype=np.uint8)])
    cloud = trimesh.PointCloud(vertices=points.astype(np.float32), colors=rgba)
    cloud.export(path)


def write_viewer(out: Path) -> None:
    viewer = out / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    (viewer / "index.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Camera-only ReSceneScribe reconstruction</title>
  <style>
    html, body { margin:0; height:100%; overflow:hidden; background:#0b0d10; color:#e8eaed; font:13px system-ui, sans-serif; }
    #info { position:absolute; left:12px; top:12px; z-index:2; background:rgba(0,0,0,.62); padding:10px 12px; border:1px solid #2d333b; max-width:420px; }
    #info strong { display:block; margin-bottom:4px; }
    canvas { display:block; }
  </style>
  <script type="importmap">{"imports":{"three":"../../../viewer/vendor/three/build/three.module.js","three/addons/":"../../../viewer/vendor/three/examples/jsm/"}}</script>
</head>
<body>
  <div id="info"><strong>Camera-only 3D reconstruction</strong><span id="status">loading scene.glb</span></div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d10);
    const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.01, 1000);
    camera.position.set(0, 12, 28);
    const renderer = new THREE.WebGLRenderer({ antialias:true });
    renderer.setSize(innerWidth, innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    document.body.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.GridHelper(80, 40, 0x38424c, 0x20262d));
    scene.add(new THREE.AxesHelper(3));
    const loader = new GLTFLoader();
    loader.load('../scene.glb', gltf => {
      scene.add(gltf.scene);
      const box = new THREE.Box3().setFromObject(gltf.scene);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(size * 0.25, size * 0.2, size * 0.45));
      camera.near = Math.max(0.01, size / 10000);
      camera.far = Math.max(1000, size * 10);
      camera.updateProjectionMatrix();
      document.getElementById('status').textContent = 'loaded RGB-only point cloud';
    }, undefined, err => {
      console.error(err);
      document.getElementById('status').textContent = 'failed to load scene.glb';
    });
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


def write_preview(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(points) > 50000:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(points), size=50000, replace=False)
        plot_points = points[idx]
        plot_colors = colors[idx] / 255.0
    else:
        plot_points = points
        plot_colors = colors / 255.0
    fig = plt.figure(figsize=(11, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(plot_points[:, 0], plot_points[:, 2], plot_points[:, 1], c=plot_colors, s=0.35, alpha=0.85)
    ax.set_xlabel("viewer x")
    ax.set_ylabel("viewer z")
    ax.set_zlabel("viewer y/up")
    ax.set_title("Camera-only RGB triangulated point cloud")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def shell(cmd: list[str], cwd: Path) -> dict:
    start = time.time()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "elapsed_s": round(time.time() - start, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RGB-only ReSceneScribe camera-only reconstruction.")
    parser.add_argument("--dataset", type=Path, default=Path("deepaccident_mini_dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs/camera_only_reconstruction"))
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--agents", nargs="+", default=AGENTS)
    parser.add_argument("--cameras", nargs="+", default=CAMERAS)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=56)
    parser.add_argument("--frame-step", type=int, default=7)
    parser.add_argument("--max-pairs", type=int, default=80)
    parser.add_argument("--max-points", type=int, default=350000)
    parser.add_argument("--voxel", type=float, default=0.08)
    parser.add_argument("--max-reproj-error", type=float, default=3.0)
    parser.add_argument("--min-angle-deg", type=float, default=0.7)
    parser.add_argument("--max-depth-m", type=float, default=160.0)
    parser.add_argument("--mask-backend", choices=["yolo", "empty"], default="yolo")
    parser.add_argument("--yolo-model", default="yolo11n-seg.pt")
    parser.add_argument("--yolo-imgsz", type=int, default=960)
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    global CATEGORY
    CATEGORY = args.category
    dataset = args.dataset.resolve()
    out = args.out.resolve()
    ensure_clean_output(out)
    frames = select_frames(args.frame_start, args.frame_end, args.frame_step)

    views = copy_inputs_and_build_views(dataset, out, args.agents, args.cameras, frames, args.scenario)
    if len(views) < 2:
        raise RuntimeError("Need at least two RGB views for reconstruction.")

    if args.mask_backend == "yolo":
        mask_report = run_yolo_masks(out, views, args.yolo_model, args.yolo_imgsz)
        if mask_report.get("status") != "ok":
            fallback = write_empty_masks(out, views)
            fallback["previous_failure"] = mask_report
            mask_report = fallback
    else:
        mask_report = write_empty_masks(out, views)

    mask_manifest = {
        "schema_version": "0.2",
        "case_id": "camera_only_reconstruction",
        "source": {
            "kind": "deepaccident_rgb_camera_frames",
            "dataset_root": str(dataset),
            "scenario": args.scenario,
            "agents": args.agents,
            "cameras": args.cameras,
            "lidar_used": False,
            "forbidden_inputs": ["lidar01", "viewer_assets/*.glb", "historical LiDAR-derived PLY/GLB"],
        },
        "frame_range": [args.frame_start, args.frame_end],
        "selected_frames": frames,
        "mask_backend": mask_report,
    }
    write_json(out / "mask_manifest.json", mask_manifest)

    reconstruction = triangulate_rgb_points(
        out=out,
        views=views,
        max_pairs=args.max_pairs,
        max_reproj_error=args.max_reproj_error,
        min_angle_deg=args.min_angle_deg,
        max_depth_m=args.max_depth_m,
    )
    points_world = reconstruction.pop("points_world")
    colors = reconstruction.pop("colors")
    if len(points_world) == 0:
        status = "failed"
        viewer_points = points_world
        origin = np.zeros(3, dtype=np.float64)
    else:
        viewer_points, origin = world_to_viewer(points_world)
        viewer_points, colors = voxel_downsample(viewer_points, colors, args.voxel, args.max_points)
        status = "ok" if len(viewer_points) >= 1000 else "weak"

    if len(viewer_points):
        write_ply(out / "reconstruction" / "points.ply", viewer_points, colors)
        export_scene_glb(out / "scene.glb", viewer_points, colors)
        write_preview(out / "reports" / "preview_point_cloud.png", viewer_points, colors)
    else:
        write_ply(out / "reconstruction" / "points.ply", np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8))

    camera_records = []
    for view in views:
        camera_records.append(
            {
                "view_id": view.view_id,
                "agent": view.agent,
                "camera": view.camera,
                "frame": view.frame,
                "rgb_path": view.rgb_rel,
                "mask_path": view.mask_rel,
                "intrinsic": view.K.tolist(),
                "camera_to_world_cv": view.c2w.tolist(),
            }
        )
    write_json(out / "reconstruction" / "cameras.json", {"views": camera_records, "world_origin_carla": origin.tolist()})

    pair_reports = reconstruction["pair_reports"]
    ok_pairs = [r for r in pair_reports if r.get("status") == "ok"]
    reproj_values = [r["median_reprojection_error_px"] for r in ok_pairs if "median_reprojection_error_px" in r]
    angle_values = [r["median_triangulation_angle_deg"] for r in ok_pairs if "median_triangulation_angle_deg" in r]
    reprojection_report = {
        "backend": reconstruction["backend"],
        "status": status,
        "input_view_count": len(views),
        "attempted_pair_count": len(pair_reports),
        "successful_pair_count": len(ok_pairs),
        "point_count": int(len(viewer_points)),
        "median_pair_reprojection_error_px": float(np.median(reproj_values)) if reproj_values else None,
        "median_pair_triangulation_angle_deg": float(np.median(angle_values)) if angle_values else None,
        "pair_reports": pair_reports,
    }
    write_json(out / "reconstruction" / "reprojection_report.json", reprojection_report)

    scale_alignment = {
        "status": "prototype_metric_prior",
        "scale_source": "DeepAccident camera pose/intrinsic calibration used as prototype camera/scale prior; no LiDAR point geometry used.",
        "residual_m": None,
        "notes": [
            "The sparse point cloud is in metric units implied by the camera pose prior.",
            "For ordinary black-box deployment, replace this with real camera calibration, lane width, surveyed scene, or known vehicle-size scale constraints.",
        ],
    }
    write_json(out / "reconstruction" / "scale_alignment.json", scale_alignment)
    write_json(out / "calibration" / "scale_constraints.json", scale_alignment)

    manifest = {
        "schema_version": "0.2",
        "case_id": "camera_only_reconstruction",
        "status": status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lidar_used": False,
        "legacy_lidar_assets_used": False,
        "forbidden_inputs": {
            "lidar01": "not read",
            "viewer_assets_glb": "not read",
            "historical_lidar_ply_glb": "not read",
        },
        "inputs": {
            "dataset_root": str(dataset),
            "scenario": args.scenario,
            "agents": args.agents,
            "cameras": args.cameras,
            "frames": frames,
            "rgb_frame_count": len(views),
            "mask_manifest": "mask_manifest.json",
        },
        "outputs": {
            "points_ply": "reconstruction/points.ply",
            "scene_glb": "scene.glb" if (out / "scene.glb").exists() else None,
            "viewer": "viewer/index.html",
            "preview": "reports/preview_point_cloud.png" if (out / "reports" / "preview_point_cloud.png").exists() else None,
            "quality_report": "reports/quality_report.md",
            "no_lidar_audit": "reports/no_lidar_audit.md",
        },
        "quality_summary": {
            "backend": reconstruction["backend"],
            "point_count": int(len(viewer_points)),
            "input_view_count": len(views),
            "successful_pair_count": len(ok_pairs),
            "median_pair_reprojection_error_px": reprojection_report["median_pair_reprojection_error_px"],
            "median_pair_triangulation_angle_deg": reprojection_report["median_pair_triangulation_angle_deg"],
        },
    }
    write_json(out / "manifest.json", manifest)

    write_viewer(out)

    py_compile = shell([sys.executable, "-m", "py_compile", str(Path(__file__).resolve())], repo)
    no_lidar_audit = f"""# No-LiDAR Audit

Status: pass for `scripts/run_camera_only_reconstruction.py`.

Command:

```bash
{' '.join(sys.argv)}
```

Input rule:

- `lidar_used=false`
- `lidar01` files were not opened by this script.
- `viewer_assets/*.glb` were not opened, copied, converted, or used as geometry input.
- Historical LiDAR-derived PLY/GLB files were not opened, copied, converted, or used as geometry input.

Inputs actually used:

- RGB images copied from `{dataset}`
- Camera intrinsic/extrinsic calibration from DeepAccident `calib` files as a prototype camera/scale prior
- Dynamic masks from `{mask_report.get('backend')}` with status `{mask_report.get('status')}`

Automated support checks:

- Script py_compile return code: `{py_compile['returncode']}`
- Output manifest records `lidar_used=false`.

Important limit:

The calibration files contain camera pose/intrinsic information and matrix names
that reference LiDAR-to-camera extrinsics in the DeepAccident calibration schema.
No LiDAR point cloud geometry from `lidar01` was consumed.
"""
    (out / "reports" / "no_lidar_audit.md").write_text(no_lidar_audit, encoding="utf-8")

    quality_report = f"""# Camera-Only Reconstruction Quality Report

Status: `{status}`

## Backend Attempt

- Backend: `{reconstruction['backend']}`
- Dynamic mask backend: `{mask_report.get('backend')}` / `{mask_report.get('status')}`
- RGB views: `{len(views)}`
- Attempted pairs: `{len(pair_reports)}`
- Successful pairs: `{len(ok_pairs)}`
- Output points after voxel/downsample: `{len(viewer_points)}`
- Median pair reprojection error px: `{reprojection_report['median_pair_reprojection_error_px']}`
- Median triangulation angle deg: `{reprojection_report['median_pair_triangulation_angle_deg']}`

## Output Artifacts

- `manifest.json`
- `mask_manifest.json`
- `reconstruction/cameras.json`
- `reconstruction/points.ply`
- `reconstruction/reprojection_report.json`
- `reconstruction/scale_alignment.json`
- `scene.glb`
- `viewer/index.html`
- `reports/preview_point_cloud.png`
- `reports/no_lidar_audit.md`

## Visual/Geometric Assessment

This is a real RGB-only sparse triangulation result, not a reused LiDAR asset.
It is expected to be much sparser than the legacy LiDAR GLB because it uses
image feature matches and camera pose priors rather than dense LiDAR returns.

Quality gate interpretation:

- `ok`: at least 1,000 filtered triangulated points and at least one successful pair.
- `weak`: some 3D points exist but density is not yet good enough.
- `failed`: no usable 3D points survived filtering.

## Fallback / Next Loop

If quality is weak, improve in this order:

1. add more RGB camera streams such as `Camera_FrontLeft` and `Camera_FrontRight`,
2. reduce `--frame-step` to include more overlap,
3. run VGGT on the same `rgb_frames` tree and export dense point maps,
4. use COLMAP or DUSt3R/MASt3R fallback if feature triangulation remains sparse,
5. review detector masks and replace weak masks with SAM 2/Grounded SAM 2 masks.
"""
    (out / "reports" / "quality_report.md").write_text(quality_report, encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0 if status in {"ok", "weak"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
