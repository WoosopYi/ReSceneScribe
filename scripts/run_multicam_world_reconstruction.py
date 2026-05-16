#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import run_camera_only_reconstruction as base


SCENARIO = base.SCENARIO
VEHICLE_AGENTS = [
    "ego_vehicle",
    "ego_vehicle_behind",
    "other_vehicle",
    "other_vehicle_behind",
]
ALL_CAMERAS = [
    "Camera_Front",
    "Camera_FrontLeft",
    "Camera_FrontRight",
    "Camera_Back",
    "Camera_BackLeft",
    "Camera_BackRight",
]
ADJACENT_CAMERA_PAIRS = {
    ("Camera_Front", "Camera_FrontLeft"),
    ("Camera_Front", "Camera_FrontRight"),
    ("Camera_FrontLeft", "Camera_BackLeft"),
    ("Camera_FrontRight", "Camera_BackRight"),
    ("Camera_Back", "Camera_BackLeft"),
    ("Camera_Back", "Camera_BackRight"),
}
AGENT_COLORS = {
    "ego_vehicle": "#ff4d4d",
    "ego_vehicle_behind": "#2dd4bf",
    "other_vehicle": "#facc15",
    "other_vehicle_behind": "#60a5fa",
}
DEFAULT_DIMENSIONS = {
    "length_m": 4.9,
    "width_m": 2.0,
    "height_m": 1.6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse four DeepAccident vehicle agents and six RGB cameras into one calibrated world-frame reconstruction."
    )
    parser.add_argument("--dataset", type=Path, default=Path("deepaccident_mini_dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs/multicam_world_reconstruction"))
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--category", default=base.DEFAULT_CATEGORY)
    parser.add_argument("--agents", nargs="+", default=VEHICLE_AGENTS)
    parser.add_argument("--cameras", nargs="+", default=ALL_CAMERAS)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=56)
    parser.add_argument("--frame-step", type=int, default=7)
    parser.add_argument("--max-pairs", type=int, default=900)
    parser.add_argument("--max-points", type=int, default=500000)
    parser.add_argument("--voxel", type=float, default=0.06)
    parser.add_argument("--max-reproj-error", type=float, default=5.0)
    parser.add_argument("--min-angle-deg", type=float, default=0.35)
    parser.add_argument("--max-depth-m", type=float, default=220.0)
    parser.add_argument("--skip-sift-reconstruction", action="store_true")
    parser.add_argument("--mask-backend", choices=["yolo", "empty"], default="yolo")
    parser.add_argument("--yolo-model", default="yolo11n-seg.pt")
    parser.add_argument("--yolo-imgsz", type=int, default=960)
    parser.add_argument("--nominal-fps", type=float, default=20.0)
    return parser.parse_args()


def clean_output(out: Path) -> None:
    base.ensure_clean_output(out)
    for rel in ["nerfstudio", "attempts"]:
        target = out / rel
        if target.exists():
            shutil.rmtree(target)
    for rel in ["nerfstudio/images", "nerfstudio/masks"]:
        (out / rel).mkdir(parents=True, exist_ok=True)


def add_pair(
    pairs: list[tuple[int, int, int, str]],
    views: list[base.View],
    ia: int,
    ib: int,
    priority: int,
    reason: str,
    seen: set[tuple[int, int]],
) -> None:
    if ia == ib:
        return
    key = (min(ia, ib), max(ia, ib))
    if key in seen:
        return
    seen.add(key)
    c1 = views[ia].c2w[:3, 3]
    c2 = views[ib].c2w[:3, 3]
    baseline = float(np.linalg.norm(c1 - c2))
    if baseline <= 0.15 or baseline > 140.0:
        return
    pairs.append((priority, key[0], key[1], reason))


def candidate_pairs_world(views: list[base.View], max_pairs: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    by_stream: dict[tuple[str, str], list[int]] = {}
    by_frame_agent: dict[tuple[int, str], list[int]] = {}
    by_frame_camera: dict[tuple[int, str], list[int]] = {}
    by_frame: dict[int, list[int]] = {}
    for idx, view in enumerate(views):
        by_stream.setdefault((view.agent, view.camera), []).append(idx)
        by_frame_agent.setdefault((view.frame, view.agent), []).append(idx)
        by_frame_camera.setdefault((view.frame, view.camera), []).append(idx)
        by_frame.setdefault(view.frame, []).append(idx)

    for idxs in by_stream.values():
        idxs = sorted(idxs, key=lambda i: views[i].frame)
        for offset, priority in [(1, 10), (2, 20), (3, 30)]:
            for ia, ib in zip(idxs, idxs[offset:]):
                add_pair(pairs, views, ia, ib, priority, f"same_stream_dt{offset}", seen)

    for idxs in by_frame_agent.values():
        idxs = sorted(idxs, key=lambda i: views[i].camera)
        by_camera = {views[i].camera: i for i in idxs}
        for cam_a, cam_b in ADJACENT_CAMERA_PAIRS:
            if cam_a in by_camera and cam_b in by_camera:
                add_pair(pairs, views, by_camera[cam_a], by_camera[cam_b], 40, "same_agent_adjacent_camera", seen)
        for pos, ia in enumerate(idxs):
            for ib in idxs[pos + 1 :]:
                add_pair(pairs, views, ia, ib, 55, "same_agent_same_frame_camera", seen)

    for idxs in by_frame_camera.values():
        idxs = sorted(idxs, key=lambda i: views[i].agent)
        for pos, ia in enumerate(idxs):
            for ib in idxs[pos + 1 :]:
                add_pair(pairs, views, ia, ib, 75, "cross_agent_same_camera_same_frame", seen)

    for frame, idxs in by_frame.items():
        frontish = [i for i in idxs if views[i].camera in {"Camera_Front", "Camera_FrontLeft", "Camera_FrontRight"}]
        for pos, ia in enumerate(sorted(frontish, key=lambda i: (views[i].agent, views[i].camera))):
            for ib in frontish[pos + 1 :]:
                if views[ia].agent == views[ib].agent:
                    continue
                if abs(frame - views[ib].frame) <= 1:
                    add_pair(pairs, views, ia, ib, 90, "cross_agent_front_overlap", seen)

    pairs.sort(key=lambda item: item[0])
    return [(ia, ib) for _priority, ia, ib, _reason in pairs[:max_pairs]]


def extract_intrinsics_for_nerfstudio(K: np.ndarray, width: int, height: int) -> dict:
    fl_x = abs(float(K[0, 1])) if abs(float(K[0, 1])) > 1.0 else abs(float(K[1, 2]))
    fl_y = abs(float(K[1, 2])) if abs(float(K[1, 2])) > 1.0 else fl_x
    cx = float(K[0, 0]) if abs(float(K[0, 0])) > 1.0 else width / 2.0
    cy = float(K[1, 0]) if abs(float(K[1, 0])) > 1.0 else height / 2.0
    return {
        "camera_model": "OPENCV",
        "w": int(width),
        "h": int(height),
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }


def write_nerfstudio_export(out: Path, views: list[base.View]) -> dict:
    image_dir = out / "nerfstudio" / "images"
    mask_dir = out / "nerfstudio" / "masks"
    # DeepAccident/CARLA camera coordinates are X forward, Y right, Z up.
    # Nerfstudio first expects an OpenCV/COLMAP camera basis before the
    # usual OpenCV -> OpenGL flip below.
    carla_cam_to_colmap_cam = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
    )
    colmap_cam_to_carla_cam = np.eye(4, dtype=np.float64)
    colmap_cam_to_carla_cam[:3, :3] = carla_cam_to_colmap_cam.T
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    frames = []
    width = height = None

    for idx, view in enumerate(views):
        stem = f"{idx:05d}__{view.agent}__{view.camera}__frame_{view.frame:03d}"
        img_dst = image_dir / f"{stem}.jpg"
        mask_dst = mask_dir / f"{stem}.png"
        shutil.copy2(out / view.rgb_rel, img_dst)
        shutil.copy2(out / view.mask_rel, mask_dst)
        with Image.open(out / view.rgb_rel) as img:
            width, height = img.size
        intr = extract_intrinsics_for_nerfstudio(view.K, int(width), int(height))
        c2w_opengl = view.c2w @ colmap_cam_to_carla_cam @ opencv_to_opengl
        record = {
            **intr,
            "file_path": f"./images/{img_dst.name}",
            "mask_path": f"./masks/{mask_dst.name}",
            "transform_matrix": c2w_opengl.tolist(),
            "source_view_id": view.view_id,
            "agent": view.agent,
            "camera": view.camera,
            "frame_index": view.frame,
        }
        frames.append(record)

    transforms = {
        "schema": "known_pose_multicam_deepaccident_v0",
        "camera_model": "OPENCV",
        "orientation_override": "none",
        "applied_transform": np.eye(3).tolist(),
        "applied_scale": 1.0,
        "aabb_scale": 32,
        "lidar_used": False,
        "notes": [
            "Camera poses come from DeepAccident calibration matrices.",
            "The calibration schema includes lidar_to_Camera names, but no lidar01 point geometry is read.",
        ],
        "frames": frames,
    }
    base.write_json(out / "nerfstudio" / "transforms.json", transforms)
    return {
        "frame_count": len(frames),
        "image_dir": "nerfstudio/images",
        "mask_dir": "nerfstudio/masks",
        "transforms": "nerfstudio/transforms.json",
        "width": width,
        "height": height,
    }


def rotation_yaw_rad(pose: np.ndarray) -> float:
    return float(math.atan2(float(pose[1, 0]), float(pose[0, 0])))


def viewer_point(world_point: np.ndarray, origin: np.ndarray) -> list[float]:
    p = np.asarray(world_point, dtype=np.float64) - origin
    return [float(p[0]), float(p[2]), float(p[1])]


def parse_label_rows(path: Path) -> tuple[list[float], list[dict]]:
    if not path.exists():
        return [], []
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    if not rows:
        return [], []
    header = []
    try:
        header = [float(v) for v in rows[0].split()[:2]]
    except ValueError:
        header = []
    objects = []
    for row in rows[1:]:
        parts = row.split()
        if len(parts) < 13:
            continue
        try:
            objects.append(
                {
                    "class_name": parts[0],
                    "position_world": [float(parts[1]), float(parts[2]), float(parts[3])],
                    "size_lwh_m": [float(parts[4]), float(parts[5]), float(parts[6])],
                    "rotation_ypr_rad": [float(parts[7]), float(parts[8]), float(parts[9])],
                    "track_id": int(float(parts[10])),
                    "aux_id": int(float(parts[11])),
                    "flag": parts[12].lower() == "true",
                }
            )
        except ValueError:
            continue
    return header, objects


def label_path(dataset: Path, agent: str, scenario: str, frame: int) -> Path:
    return base.dataset_path(dataset, agent, "label", scenario) / f"{scenario}_{frame:03d}.txt"


def ego_dimensions_from_labels(dataset: Path, agent: str, scenario: str, frames: list[int]) -> dict:
    dims = []
    for frame in frames:
        _header, objects = parse_label_rows(label_path(dataset, agent, scenario, frame))
        for obj in objects:
            if obj["track_id"] == -100:
                dims.append(obj["size_lwh_m"])
                break
    if not dims:
        return dict(DEFAULT_DIMENSIONS)
    arr = np.asarray(dims, dtype=np.float64)
    med = np.median(arr, axis=0)
    return {
        "length_m": float(med[0]),
        "width_m": float(med[1]),
        "height_m": float(med[2]),
    }


def build_agent_tracks(
    dataset: Path,
    out: Path,
    agents: list[str],
    frames: list[int],
    scenario: str,
    origin: np.ndarray,
    nominal_fps: float,
) -> dict:
    tracks = {}
    dimensions = {}
    for agent in agents:
        dimensions[agent] = ego_dimensions_from_labels(dataset, agent, scenario, frames)
        samples = []
        for frame in frames:
            calib = base.load_calib(dataset, agent, frame, scenario)
            pose = np.asarray(calib["ego_to_world"], dtype=np.float64)
            position = pose[:3, 3].astype(np.float64)
            samples.append(
                {
                    "frame": int(frame),
                    "position_world": position.tolist(),
                    "position_viewer": viewer_point(position, origin),
                    "yaw_rad": rotation_yaw_rad(pose),
                    "ego_to_world": pose.tolist(),
                }
            )
        for i, sample in enumerate(samples):
            if i == 0:
                sample["velocity_m_per_frame"] = [0.0, 0.0, 0.0]
                sample["speed_m_per_frame"] = 0.0
                sample["nominal_speed_mps"] = 0.0
                continue
            prev = samples[i - 1]
            frame_delta = max(1, int(sample["frame"]) - int(prev["frame"]))
            delta = (np.asarray(sample["position_world"]) - np.asarray(prev["position_world"])) / float(frame_delta)
            sample["velocity_m_per_frame"] = delta.tolist()
            sample["speed_m_per_frame"] = float(np.linalg.norm(delta))
            sample["nominal_speed_mps"] = float(np.linalg.norm(delta) * nominal_fps)
        tracks[agent] = {
            "color": AGENT_COLORS.get(agent, "#ffffff"),
            "dimensions": dimensions[agent],
            "samples": samples,
        }

    viewer_tracks = {
        "schema_version": "0.1",
        "coordinate_system": "viewer xyz = world x,z,y minus reconstruction origin",
        "world_origin": origin.tolist(),
        "agents": tracks,
    }
    base.write_json(out / "replay" / "agent_tracks.json", viewer_tracks)
    return viewer_tracks


def build_object_tracks(
    dataset: Path,
    out: Path,
    agents: list[str],
    frames: list[int],
    scenario: str,
) -> dict:
    records = []
    flagged = []
    for agent in agents:
        for frame in frames:
            header, objects = parse_label_rows(label_path(dataset, agent, scenario, frame))
            for obj in objects:
                record = {
                    "observer_agent": agent,
                    "frame": int(frame),
                    **obj,
                }
                records.append(record)
                if obj.get("flag"):
                    flagged.append(record)
    payload = {
        "schema_version": "0.1",
        "source": "DeepAccident label files for object-context only",
        "label_header_note": "The first row is preserved as uninterpreted label metadata.",
        "record_count": len(records),
        "flagged_record_count": len(flagged),
        "flagged_records": flagged[:80],
    }
    base.write_json(out / "replay" / "object_context_from_labels.json", payload)
    return payload


def proxy_radius(dimensions: dict) -> float:
    length = float(dimensions.get("length_m", DEFAULT_DIMENSIONS["length_m"]))
    width = float(dimensions.get("width_m", DEFAULT_DIMENSIONS["width_m"]))
    return 0.5 * math.sqrt(length * length + width * width)


def accident_diagnostics(out: Path, tracks: dict) -> dict:
    agents = list(tracks["agents"].keys())
    per_frame = {}
    for agent in agents:
        for sample in tracks["agents"][agent]["samples"]:
            per_frame.setdefault(int(sample["frame"]), {})[agent] = sample

    pair_records = []
    for frame, samples_by_agent in sorted(per_frame.items()):
        for i, agent_a in enumerate(agents):
            if agent_a not in samples_by_agent:
                continue
            for agent_b in agents[i + 1 :]:
                if agent_b not in samples_by_agent:
                    continue
                pa = np.asarray(samples_by_agent[agent_a]["position_world"], dtype=np.float64)
                pb = np.asarray(samples_by_agent[agent_b]["position_world"], dtype=np.float64)
                d3 = float(np.linalg.norm(pa - pb))
                dxy = float(np.linalg.norm(pa[:2] - pb[:2]))
                clearance = dxy - proxy_radius(tracks["agents"][agent_a]["dimensions"]) - proxy_radius(
                    tracks["agents"][agent_b]["dimensions"]
                )
                pair_records.append(
                    {
                        "frame": int(frame),
                        "agents": [agent_a, agent_b],
                        "center_distance_3d_m": d3,
                        "center_distance_xy_m": dxy,
                        "proxy_clearance_xy_m": float(clearance),
                    }
                )

    closest = min(pair_records, key=lambda r: r["proxy_clearance_xy_m"]) if pair_records else None
    sequence = []
    if closest is not None:
        pair = closest["agents"]
        series = [r for r in pair_records if r["agents"] == pair]
        series = sorted(series, key=lambda r: r["frame"])
        idx = next(i for i, r in enumerate(series) if r["frame"] == closest["frame"])
        pre = series[max(0, idx - 2)] if series else None
        post = series[min(len(series) - 1, idx + 2)] if series else None
        sequence = [
            {
                "phase": "pre_closest",
                "frame": pre["frame"] if pre else None,
                "proxy_clearance_xy_m": pre["proxy_clearance_xy_m"] if pre else None,
            },
            {
                "phase": "closest",
                "frame": closest["frame"],
                "proxy_clearance_xy_m": closest["proxy_clearance_xy_m"],
            },
            {
                "phase": "post_closest",
                "frame": post["frame"] if post else None,
                "proxy_clearance_xy_m": post["proxy_clearance_xy_m"] if post else None,
            },
        ]
    payload = {
        "schema_version": "0.1",
        "method": "calibration_pose_center_distance_with_proxy_vehicle_radii",
        "proxy_only": True,
        "warning": "This is a scene reconstruction diagnostic, not legal or physical crash proof.",
        "closest_approach": closest,
        "sequence_summary": sequence,
        "pair_records": pair_records,
    }
    base.write_json(out / "replay" / "accident_diagnostics.json", payload)
    return payload


def write_viewer(out: Path) -> None:
    viewer = out / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    (viewer / "index.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Multi-camera world reconstruction</title>
  <style>
    html, body { margin:0; height:100%; overflow:hidden; background:#07090c; color:#e8eaed; font:13px system-ui, sans-serif; }
    #hud { position:absolute; left:12px; top:12px; z-index:2; width:min(440px, calc(100vw - 24px)); background:rgba(7,9,12,.74); border:1px solid #2d333b; padding:10px 12px; box-sizing:border-box; }
    #hud strong { display:block; margin-bottom:4px; font-size:14px; }
    #hud .row { display:flex; justify-content:space-between; gap:12px; white-space:nowrap; }
    #hud input { width:100%; margin-top:8px; }
    #legend { display:grid; grid-template-columns:1fr 1fr; gap:4px 10px; margin-top:8px; }
    .swatch { display:inline-block; width:10px; height:10px; margin-right:6px; border-radius:50%; vertical-align:-1px; }
    canvas { display:block; }
  </style>
  <script type="importmap">{"imports":{"three":"../../../viewer/vendor/three/build/three.module.js","three/addons/":"../../../viewer/vendor/three/examples/jsm/"}}</script>
</head>
<body>
  <div id="hud">
    <strong>Multi-agent camera-only world reconstruction</strong>
    <div class="row"><span id="status">loading</span><span id="frameLabel"></span></div>
    <input id="frame" type="range" min="0" max="0" value="0" step="1" />
    <div id="closest"></div>
    <div id="legend"></div>
  </div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07090c);
    const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.01, 2000);
    camera.position.set(0, 18, 45);
    const renderer = new THREE.WebGLRenderer({ antialias:true });
    renderer.setSize(innerWidth, innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    document.body.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.GridHelper(120, 60, 0x38424c, 0x20262d));
    scene.add(new THREE.AxesHelper(4));
    scene.add(new THREE.HemisphereLight(0xffffff, 0x1b2430, 1.15));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.45);
    keyLight.position.set(18, 36, 24);
    scene.add(keyLight);

    const statusEl = document.getElementById('status');
    const frameEl = document.getElementById('frame');
    const frameLabelEl = document.getElementById('frameLabel');
    const closestEl = document.getElementById('closest');
    const legendEl = document.getElementById('legend');
    const dynamicGroup = new THREE.Group();
    scene.add(dynamicGroup);
    let tracks = null;
    let diagnostics = null;
    let frames = [];
    let markers = [];

    function hexToNumber(hex) {
      return Number.parseInt(hex.replace('#', ''), 16);
    }

    function makeLine(points, color) {
      const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
      const material = new THREE.LineBasicMaterial({ color, linewidth: 2 });
      return new THREE.Line(geometry, material);
    }

    function makeCarMarker(dimensions, color) {
      const length = Math.max(2.8, dimensions.length_m || 4.5);
      const width = Math.max(1.4, dimensions.width_m || 1.9);
      const height = Math.max(1.2, dimensions.height_m || 1.5);
      const group = new THREE.Group();
      const bodyHeight = height * 0.38;
      const cabinHeight = height * 0.36;
      const wheelRadius = Math.max(0.28, height * 0.18);
      const wheelDepth = Math.max(0.18, width * 0.14);
      const y0 = -height * 0.5;
      const bodyMaterial = new THREE.MeshStandardMaterial({ color, roughness:0.42, metalness:0.22 });
      const trimMaterial = new THREE.MeshStandardMaterial({ color:0x232a32, roughness:0.38, metalness:0.12 });
      const darkMaterial = new THREE.MeshStandardMaterial({ color:0x111820, roughness:0.35, metalness:0.05 });
      const glassMaterial = new THREE.MeshStandardMaterial({ color:0x8fb8d8, roughness:0.18, metalness:0.02, transparent:true, opacity:0.72 });
      const wheelMaterial = new THREE.MeshStandardMaterial({ color:0x050607, roughness:0.65, metalness:0.05 });
      const rimMaterial = new THREE.MeshStandardMaterial({ color:0xb6bcc6, roughness:0.34, metalness:0.55 });
      const lightMaterial = new THREE.MeshStandardMaterial({ color:0xfff1b0, emissive:0xffdf7a, emissiveIntensity:0.55 });
      const tailMaterial = new THREE.MeshStandardMaterial({ color:0xff2536, emissive:0xa00018, emissiveIntensity:0.45 });

      const body = new THREE.Mesh(new THREE.BoxGeometry(length * 0.82, bodyHeight, width * 0.94), bodyMaterial);
      body.position.y = y0 + wheelRadius + bodyHeight * 0.55;
      group.add(body);
      const hood = new THREE.Mesh(new THREE.BoxGeometry(length * 0.24, bodyHeight * 0.45, width * 0.86), bodyMaterial);
      hood.position.set(length * 0.23, y0 + wheelRadius + bodyHeight * 0.92, 0);
      group.add(hood);

      const trunk = new THREE.Mesh(new THREE.BoxGeometry(length * 0.18, bodyHeight * 0.38, width * 0.82), bodyMaterial);
      trunk.position.set(-length * 0.30, y0 + wheelRadius + bodyHeight * 0.86, 0);
      group.add(trunk);

      const cabin = new THREE.Mesh(new THREE.BoxGeometry(length * 0.34, cabinHeight, width * 0.58), glassMaterial);
      cabin.position.set(-length * 0.04, y0 + wheelRadius + bodyHeight + cabinHeight * 0.52, 0);
      group.add(cabin);

      const roof = new THREE.Mesh(new THREE.BoxGeometry(length * 0.24, cabinHeight * 0.12, width * 0.50), bodyMaterial);
      roof.position.set(-length * 0.05, y0 + wheelRadius + bodyHeight + cabinHeight * 1.08, 0);
      group.add(roof);

      const windshield = new THREE.Mesh(new THREE.BoxGeometry(length * 0.035, cabinHeight * 0.70, width * 0.54), glassMaterial);
      windshield.position.set(length * 0.13, y0 + wheelRadius + bodyHeight + cabinHeight * 0.45, 0);
      windshield.rotation.z = -0.36;
      group.add(windshield);

      const rearWindow = new THREE.Mesh(new THREE.BoxGeometry(length * 0.035, cabinHeight * 0.62, width * 0.52), glassMaterial);
      rearWindow.position.set(-length * 0.22, y0 + wheelRadius + bodyHeight + cabinHeight * 0.43, 0);
      rearWindow.rotation.z = 0.30;
      group.add(rearWindow);

      for (const z of [-width * 0.31, width * 0.31]) {
        const sideWindow = new THREE.Mesh(new THREE.BoxGeometry(length * 0.26, cabinHeight * 0.45, width * 0.025), glassMaterial);
        sideWindow.position.set(-length * 0.04, y0 + wheelRadius + bodyHeight + cabinHeight * 0.58, z);
        group.add(sideWindow);
        const sideTrim = new THREE.Mesh(new THREE.BoxGeometry(length * 0.70, bodyHeight * 0.10, width * 0.025), trimMaterial);
        sideTrim.position.set(-length * 0.02, y0 + wheelRadius + bodyHeight * 0.86, z * 1.04);
        group.add(sideTrim);
      }

      const bumperFront = new THREE.Mesh(new THREE.BoxGeometry(length * 0.055, bodyHeight * 0.26, width * 0.82), darkMaterial);
      bumperFront.position.set(length * 0.455, y0 + wheelRadius + bodyHeight * 0.34, 0);
      group.add(bumperFront);
      const bumperRear = bumperFront.clone();
      bumperRear.position.x = -length * 0.455;
      group.add(bumperRear);

      const wheelGeometry = new THREE.CylinderGeometry(wheelRadius, wheelRadius, wheelDepth, 24);
      const rimGeometry = new THREE.CylinderGeometry(wheelRadius * 0.48, wheelRadius * 0.48, wheelDepth * 1.06, 18);
      for (const x of [-length * 0.30, length * 0.30]) {
        for (const z of [-width * 0.52, width * 0.52]) {
          const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
          wheel.rotation.x = Math.PI / 2;
          wheel.position.set(x, y0 + wheelRadius, z);
          group.add(wheel);
          const rim = new THREE.Mesh(rimGeometry, rimMaterial);
          rim.rotation.x = Math.PI / 2;
          rim.position.copy(wheel.position);
          group.add(rim);
        }
      }

      const grille = new THREE.Mesh(new THREE.BoxGeometry(length * 0.025, bodyHeight * 0.22, width * 0.32), darkMaterial);
      grille.position.set(length * 0.485, y0 + wheelRadius + bodyHeight * 0.58, 0);
      group.add(grille);

      for (const z of [-width * 0.24, width * 0.24]) {
        const headlight = new THREE.Mesh(new THREE.BoxGeometry(length * 0.025, bodyHeight * 0.16, width * 0.15), lightMaterial);
        headlight.position.set(length * 0.49, y0 + wheelRadius + bodyHeight * 0.60, z);
        group.add(headlight);
        const tail = new THREE.Mesh(new THREE.BoxGeometry(length * 0.025, bodyHeight * 0.18, width * 0.13), tailMaterial);
        tail.position.set(-length * 0.49, y0 + wheelRadius + bodyHeight * 0.58, z);
        group.add(tail);
      }
      return group;
    }

    function addTracks() {
      if (!tracks) return;
      legendEl.innerHTML = '';
      for (const [agent, data] of Object.entries(tracks.agents)) {
        const color = hexToNumber(data.color || '#ffffff');
        const points = data.samples.map(s => s.position_viewer);
        scene.add(makeLine(points, color));
        const div = document.createElement('div');
        div.innerHTML = `<span class="swatch" style="background:${data.color}"></span>${agent}`;
        legendEl.appendChild(div);
        const marker = makeCarMarker(data.dimensions, color);
        marker.userData.agent = agent;
        dynamicGroup.add(marker);
        markers.push(marker);
      }
      frames = [...new Set(Object.values(tracks.agents).flatMap(d => d.samples.map(s => s.frame)))].sort((a, b) => a - b);
      frameEl.max = Math.max(0, frames.length - 1);
      updateFrame(0);
    }

    function updateFrame(idx) {
      if (!tracks || frames.length === 0) return;
      const frame = frames[idx];
      frameLabelEl.textContent = `frame ${frame}`;
      markers.forEach(marker => {
        const sample = tracks.agents[marker.userData.agent].samples.find(s => s.frame === frame);
        if (!sample) {
          marker.visible = false;
          return;
        }
        marker.visible = true;
        marker.position.set(sample.position_viewer[0], sample.position_viewer[1], sample.position_viewer[2]);
        marker.rotation.y = -sample.yaw_rad;
      });
    }

    frameEl.addEventListener('input', () => updateFrame(Number(frameEl.value)));

    const loader = new GLTFLoader();
    loader.load('../scene.glb', gltf => {
      scene.add(gltf.scene);
      const box = new THREE.Box3().setFromObject(gltf.scene);
      const center = box.getCenter(new THREE.Vector3());
      const size = Math.max(10, box.getSize(new THREE.Vector3()).length());
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(size * 0.25, size * 0.18, size * 0.42));
      camera.near = Math.max(0.01, size / 10000);
      camera.far = Math.max(2000, size * 10);
      camera.updateProjectionMatrix();
      statusEl.textContent = 'scene.glb loaded';
    }, undefined, err => {
      console.error(err);
      statusEl.textContent = 'scene.glb failed';
    });

    Promise.all([
      fetch('../replay/agent_tracks.json').then(r => r.json()),
      fetch('../replay/accident_diagnostics.json').then(r => r.json())
    ]).then(([trackJson, diagJson]) => {
      tracks = trackJson;
      diagnostics = diagJson;
      addTracks();
      const c = diagnostics.closest_approach;
      if (c) closestEl.textContent = `closest proxy: ${c.agents.join(' / ')} frame ${c.frame}, clearance ${c.proxy_clearance_xy_m.toFixed(2)}m`;
    }).catch(err => {
      console.error(err);
      closestEl.textContent = 'track JSON failed';
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


def write_reports(
    out: Path,
    args: argparse.Namespace,
    views: list[base.View],
    frames: list[int],
    reconstruction: dict,
    reprojection_report: dict,
    mask_report: dict,
    ns_report: dict,
    diagnostics: dict,
    status: str,
) -> None:
    ok_pairs = [r for r in reconstruction["pair_reports"] if r.get("status") == "ok"]
    no_lidar = f"""# No-LiDAR Audit

Status: pass for `scripts/run_multicam_world_reconstruction.py`.

Command:

```bash
{' '.join(sys.argv)}
```

Inputs used:

- RGB images from `{args.dataset}`
- DeepAccident calibration matrices for camera and ego poses
- Dynamic masks from `{mask_report.get('backend')}` with status `{mask_report.get('status')}`
- Label text files for object context only

Inputs not used:

- `lidar01` point clouds: not read
- `viewer_assets/*.glb`: not read
- Historical LiDAR-derived PLY/GLB: not read

Calibration note:

The DeepAccident calibration schema names some matrices `lidar_to_Camera_*`.
This script uses those matrices only as camera extrinsic calibration chains.
It never reads LiDAR point geometry and records `lidar_used=false` in the manifest.
"""
    (out / "reports" / "no_lidar_audit.md").write_text(no_lidar, encoding="utf-8")

    closest = diagnostics.get("closest_approach")
    quality = f"""# Multi-camera World Reconstruction Quality Report

Status: `{status}`

## Inputs

- Vehicle agents: `{', '.join(args.agents)}`
- Cameras: `{', '.join(args.cameras)}`
- Selected frames: `{frames[0]}` to `{frames[-1]}` step `{args.frame_step}` (`{len(frames)}` frames)
- RGB views: `{len(views)}`
- Nerfstudio frames: `{ns_report['frame_count']}`
- LiDAR geometry used: `false`

## Reconstruction Metrics

- Backend: `{reconstruction['backend']}`
- Attempted pairs: `{len(reconstruction['pair_reports'])}`
- Successful pairs: `{len(ok_pairs)}`
- Output points: `{reprojection_report['point_count']}`
- Median pair reprojection error px: `{reprojection_report['median_pair_reprojection_error_px']}`
- Median triangulation angle deg: `{reprojection_report['median_pair_triangulation_angle_deg']}`
- Mask backend: `{mask_report.get('backend')}` / `{mask_report.get('status')}`

## Motion And Accident Context

- Track source: `ego_to_world` calibration pose for each vehicle agent and frame.
- Closest proxy approach: `{closest}`
- Proxy caveat: closest approach uses center distances and approximate vehicle radii, not full physics contact simulation.

## Output Artifacts

- `manifest.json`
- `nerfstudio/transforms.json`
- `reconstruction/points.ply`
- `reconstruction/reprojection_report.json`
- `replay/agent_tracks.json`
- `replay/accident_diagnostics.json`
- `viewer/index.html`
- `scene.glb`

## Quality Loop Notes

This run fuses all selected cameras into one world coordinate system. If the point
cloud is still sparse, the next loop should reduce `--frame-step`, increase
`--max-pairs`, or run a NeRF/3DGS backend from `nerfstudio/transforms.json`.
"""
    (out / "reports" / "quality_report.md").write_text(quality, encoding="utf-8")

    ns_status = f"""# Nerfstudio Export Status

Status: exported known-pose dataset.

- Transforms: `nerfstudio/transforms.json`
- Images: `nerfstudio/images/`
- Masks: `nerfstudio/masks/`
- Frames: `{ns_report['frame_count']}`
- Image size: `{ns_report['width']}x{ns_report['height']}`

Suggested command:

```bash
ns-train nerfacto --data {out / 'nerfstudio'} --pipeline.model.background-color black
```

The export is the handoff artifact for Nerfstudio or instant-ngp. It keeps all
four vehicle-agent camera rigs in one calibration-derived world frame.
"""
    (out / "reports" / "nerfstudio_status.md").write_text(ns_status, encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = Path.cwd().resolve()
    base.CATEGORY = args.category
    dataset = args.dataset.resolve()
    out = args.out.resolve()
    clean_output(out)
    frames = base.select_frames(args.frame_start, args.frame_end, args.frame_step)

    views = base.copy_inputs_and_build_views(dataset, out, args.agents, args.cameras, frames, args.scenario)
    if len(views) < 2:
        raise RuntimeError("Need at least two RGB views for multi-camera reconstruction.")

    if args.mask_backend == "yolo":
        mask_report = base.run_yolo_masks(out, views, args.yolo_model, args.yolo_imgsz)
        if mask_report.get("status") != "ok":
            fallback = base.write_empty_masks(out, views)
            fallback["previous_failure"] = mask_report
            mask_report = fallback
    else:
        mask_report = base.write_empty_masks(out, views)

    mask_manifest = {
        "schema_version": "0.3",
        "case_id": "multicam_world_reconstruction",
        "source": {
            "kind": "deepaccident_rgb_camera_frames",
            "dataset_root": str(dataset),
            "scenario": args.scenario,
            "category": args.category,
            "agents": args.agents,
            "cameras": args.cameras,
            "lidar_used": False,
            "forbidden_inputs": ["lidar01", "viewer_assets/*.glb", "historical LiDAR-derived PLY/GLB"],
        },
        "frame_range": [args.frame_start, args.frame_end],
        "selected_frames": frames,
        "mask_backend": mask_report,
    }
    base.write_json(out / "mask_manifest.json", mask_manifest)

    ns_report = write_nerfstudio_export(out, views)
    if args.skip_sift_reconstruction:
        reconstruction = {
            "backend": "export_only_known_pose_dataset_no_sift",
            "status": "skipped",
            "pair_reports": [],
            "points_world": np.zeros((0, 3), dtype=np.float64),
            "colors": np.zeros((0, 3), dtype=np.uint8),
        }
    else:
        base.candidate_pairs = candidate_pairs_world
        reconstruction = base.triangulate_rgb_points(
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
        status = "export_only" if args.skip_sift_reconstruction else "failed"
        viewer_points = points_world
        origin = np.zeros(3, dtype=np.float64)
    else:
        viewer_points, origin = base.world_to_viewer(points_world)
        viewer_points, colors = base.voxel_downsample(viewer_points, colors, args.voxel, args.max_points)
        status = "ok" if len(viewer_points) >= 5000 else "weak"

    if len(viewer_points):
        base.write_ply(out / "reconstruction" / "points.ply", viewer_points, colors)
        base.export_scene_glb(out / "scene.glb", viewer_points, colors)
        base.write_preview(out / "reports" / "preview_point_cloud.png", viewer_points, colors)
    else:
        base.write_ply(
            out / "reconstruction" / "points.ply",
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint8),
        )

    tracks = build_agent_tracks(dataset, out, args.agents, frames, args.scenario, origin, args.nominal_fps)
    object_context = build_object_tracks(dataset, out, args.agents, frames, args.scenario)
    diagnostics = accident_diagnostics(out, tracks)

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
                "intrinsic_raw": view.K.tolist(),
                "camera_to_world_cv": view.c2w.tolist(),
            }
        )
    base.write_json(
        out / "reconstruction" / "cameras.json",
        {
            "views": camera_records,
            "world_origin_carla": origin.tolist(),
            "coordinate_note": "Calibration-derived world frame; viewer maps world x,z,y after origin subtraction.",
        },
    )

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
    base.write_json(out / "reconstruction" / "reprojection_report.json", reprojection_report)
    base.write_json(
        out / "reconstruction" / "scale_alignment.json",
        {
            "status": "calibration_metric_prior",
            "scale_source": "DeepAccident camera and ego pose calibration. No lidar01 geometry.",
            "world_origin_carla": origin.tolist(),
            "residual_m": None,
        },
    )

    write_viewer(out)
    py_compile = base.shell([sys.executable, "-m", "py_compile", str(Path(__file__).resolve())], repo)
    write_reports(out, args, views, frames, reconstruction, reprojection_report, mask_report, ns_report, diagnostics, status)

    manifest = {
        "schema_version": "0.3",
        "case_id": "multicam_world_reconstruction",
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
            "category": args.category,
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
            "nerfstudio_transforms": "nerfstudio/transforms.json",
            "agent_tracks": "replay/agent_tracks.json",
            "accident_diagnostics": "replay/accident_diagnostics.json",
            "object_context": "replay/object_context_from_labels.json",
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
            "py_compile_returncode": py_compile["returncode"],
            "object_context_records": object_context["record_count"],
        },
    }
    base.write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0 if status in {"ok", "weak", "export_only"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
