#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from plyfile import PlyData

import run_camera_only_reconstruction as base
import run_multicam_world_reconstruction as multi
import run_slam3r_deepaccident_reconstruction as slam3r


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
class LayerSpec:
    name: str
    description: str
    point_conf_percentile: float
    points_per_stream: int
    require_mask_filter: bool
    disable_mask_filter: bool
    max_camera_depth: float
    max_metric_depth_m: float
    max_frame_alignment_error_m: float
    max_stream_rmse_m: float
    max_stream_median_error_m: float
    max_stream_max_error_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build additive no-LiDAR SLAM3R reconstruction layers from the current good base."
    )
    parser.add_argument("--dataset", type=Path, default=Path("deepaccident_mini_dataset"))
    parser.add_argument("--source", type=Path, default=Path("outputs/town04_type1_subtype2_slam3r_reconstruction"))
    parser.add_argument("--out", type=Path, default=Path("outputs/town04_type1_subtype2_slam3r_incremental_layers"))
    parser.add_argument("--mask-export", type=Path, default=Path("outputs/town04_type1_subtype2_multicam_export"))
    parser.add_argument("--slam3r-root", type=Path, default=Path("third_party/SLAM3R"))
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--agents", nargs="+", default=AGENTS)
    parser.add_argument("--cameras", nargs="+", default=CAMERAS)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=49)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--voxel", type=float, default=0.08)
    parser.add_argument("--add-voxel", type=float, default=0.08)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Optional per-layer/final cap. The default 0 preserves the base and all accepted additive voxels.",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--nominal-fps", type=float, default=20.0)
    parser.add_argument("--min-world-z", type=float, default=-3.0)
    parser.add_argument("--max-world-z", type=float, default=25.0)
    return parser.parse_args()


def clean_output(out: Path) -> None:
    if out.exists():
        for rel in ["layers", "stages", "reconstruction", "reports", "replay", "viewer"]:
            target = out / rel
            if target.exists():
                shutil.rmtree(target)
        for rel in ["scene.glb", "manifest.json", "summary.json"]:
            target = out / rel
            if target.exists():
                target.unlink()
    for rel in ["layers", "stages", "reconstruction", "reports", "replay", "viewer"]:
        (out / rel).mkdir(parents=True, exist_ok=True)


def read_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    colors = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
    return points, colors


def make_streams(args: argparse.Namespace) -> list[slam3r.Stream]:
    requested_frames = base.select_frames(args.frame_start, args.frame_end, args.frame_step, include_end=True)
    streams: list[slam3r.Stream] = []
    for agent in args.agents:
        for camera in args.cameras:
            frames = slam3r.select_existing_frames(args.dataset, agent, camera, args.scenario, requested_frames)
            if not frames:
                continue
            streams.append(
                slam3r.Stream(
                    agent=agent,
                    camera=camera,
                    frames=frames,
                    input_dir=args.source / "input_streams" / agent / camera,
                    test_name=f"{agent}__{camera}",
                    gpu_id=0,
                )
            )
    if not streams:
        raise RuntimeError("No streams found for incremental layering.")
    return streams


def layer_specs() -> list[LayerSpec]:
    return [
        LayerSpec(
            name="01_masked_dense",
            description="Same accepted streams as base, masked frames only, lower confidence cutoff for denser static background.",
            point_conf_percentile=50.0,
            points_per_stream=160_000,
            require_mask_filter=True,
            disable_mask_filter=False,
            max_camera_depth=4.8,
            max_metric_depth_m=90.0,
            max_frame_alignment_error_m=4.0,
            max_stream_rmse_m=2.5,
            max_stream_median_error_m=1.8,
            max_stream_max_error_m=7.0,
        ),
        LayerSpec(
            name="02_all_aligned_frames",
            description="Accepted streams plus additional aligned frames; masks are used where present, otherwise high-confidence RGB depth only.",
            point_conf_percentile=68.0,
            points_per_stream=110_000,
            require_mask_filter=False,
            disable_mask_filter=False,
            max_camera_depth=4.0,
            max_metric_depth_m=75.0,
            max_frame_alignment_error_m=3.0,
            max_stream_rmse_m=2.5,
            max_stream_median_error_m=1.8,
            max_stream_max_error_m=7.0,
        ),
        LayerSpec(
            name="03_strict_extra_streams",
            description="Relax stream-level rejection but keep strict per-frame alignment, masks, and high confidence.",
            point_conf_percentile=75.0,
            points_per_stream=80_000,
            require_mask_filter=True,
            disable_mask_filter=False,
            max_camera_depth=3.8,
            max_metric_depth_m=70.0,
            max_frame_alignment_error_m=2.0,
            max_stream_rmse_m=4.5,
            max_stream_median_error_m=3.0,
            max_stream_max_error_m=25.0,
        ),
    ]


def make_fuse_args(args: argparse.Namespace, spec: LayerSpec) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=args.dataset,
        out=args.source,
        slam3r_root=args.slam3r_root,
        scenario=args.scenario,
        category=args.category,
        seed=args.seed,
        point_conf_percentile=spec.point_conf_percentile,
        pose_conf_percentile=70.0,
        fusion_mode="camera-ray-depth",
        mask_export=args.mask_export,
        disable_mask_filter=spec.disable_mask_filter,
        require_mask_filter=spec.require_mask_filter,
        min_camera_depth=0.05,
        max_camera_depth=spec.max_camera_depth,
        max_metric_depth_m=spec.max_metric_depth_m,
        depth_trim_percentile=1.0,
        disable_sky_filter=False,
        max_stream_rmse_m=spec.max_stream_rmse_m,
        max_stream_median_error_m=spec.max_stream_median_error_m,
        max_stream_max_error_m=spec.max_stream_max_error_m,
        max_frame_alignment_error_m=spec.max_frame_alignment_error_m,
        min_world_z=args.min_world_z,
        max_world_z=args.max_world_z,
        trim_percentile=1.0,
        points_per_stream=spec.points_per_stream,
        max_points=args.max_points,
        voxel=args.voxel,
    )


def select_new_voxels(
    points: np.ndarray,
    colors: np.ndarray,
    occupied: set[tuple[int, int, int]],
    add_voxel: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(points) == 0:
        return points, colors, 0
    keys = np.floor(points / add_voxel).astype(np.int64)
    keep = np.zeros(len(points), dtype=bool)
    duplicates = 0
    for idx, key_arr in enumerate(keys):
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
        if key in occupied:
            duplicates += 1
            continue
        occupied.add(key)
        keep[idx] = True
    return points[keep], colors[keep], duplicates


def occupied_keys(points: np.ndarray, voxel: float) -> set[tuple[int, int, int]]:
    keys = np.floor(points / voxel).astype(np.int64)
    return {(int(k[0]), int(k[1]), int(k[2])) for k in keys}


def write_topdown(path: Path, points: np.ndarray, colors: np.ndarray, title: str, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = np.arange(len(points))
    if len(idx) > 180_000:
        idx = np.sort(rng.choice(idx, size=180_000, replace=False))
    pts = points[idx]
    rgb = colors[idx] / 255.0
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.scatter(pts[:, 0], pts[:, 1], c=rgb, s=0.08, linewidths=0, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_cloud_bundle(
    out: Path,
    points_world: np.ndarray,
    colors: np.ndarray,
    origin: np.ndarray,
    title: str,
    seed: int,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "reconstruction").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    viewer_points, _ = base.world_to_viewer(points_world, origin=origin)
    base.write_ply(out / "reconstruction" / "points_world.ply", points_world, colors)
    base.write_ply(out / "reconstruction" / "points.ply", viewer_points, colors)
    base.export_scene_glb(out / "scene.glb", viewer_points, colors)
    base.write_preview(out / "reports" / "preview_point_cloud.png", viewer_points, colors)
    write_topdown(out / "reports" / "topdown_preview.png", points_world, colors, title, seed)
    return {
        "point_count": int(len(points_world)),
        "bbox_min_world": [float(x) for x in points_world.min(axis=0)] if len(points_world) else None,
        "bbox_max_world": [float(x) for x in points_world.max(axis=0)] if len(points_world) else None,
        "scene_glb": str(out / "scene.glb"),
        "points_world_ply": str(out / "reconstruction" / "points_world.ply"),
    }


def write_incremental_viewer(out: Path, stage_reports: list[dict], layer_reports: list[dict]) -> None:
    viewer = out / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    stages = [
        {
            "id": item["stage"],
            "label": f"{item['stage']} ({item['cumulative_points']:,} pts)",
            "url": f"../stages/{item['stage']}/scene.glb",
            "points": int(item["cumulative_points"]),
        }
        for item in stage_reports
    ]
    layers = [
        {
            "id": item["layer"],
            "label": f"{item['layer']} (+{item['new_points']:,} pts)",
            "url": f"../layers/{item['layer']}/scene.glb",
            "points": int(item["new_points"]),
        }
        for item in layer_reports
    ]
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Incremental no-LiDAR reconstruction</title>
  <style>
    html, body { margin:0; height:100%; overflow:hidden; background:#07090c; color:#e8eaed; font:13px system-ui, sans-serif; }
    canvas { display:block; }
    #hud { position:absolute; left:12px; top:12px; z-index:2; width:min(520px, calc(100vw - 24px)); background:rgba(7,9,12,.78); border:1px solid #2d333b; padding:10px 12px; box-sizing:border-box; }
    #hud strong { display:block; margin-bottom:6px; font-size:14px; }
    #hud .row { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:6px 0; }
    #hud label { display:flex; align-items:center; gap:8px; min-width:0; }
    #stageSelect { flex:1; min-width:160px; background:#11161d; color:#e8eaed; border:1px solid #38424c; padding:4px 6px; }
    #frame { width:100%; margin-top:6px; }
    #layers { display:grid; grid-template-columns:1fr; gap:4px; margin:8px 0; }
    #layers label { justify-content:flex-start; }
    #legend { display:grid; grid-template-columns:1fr 1fr; gap:4px 10px; margin-top:8px; }
    .swatch { display:inline-block; width:10px; height:10px; margin-right:6px; border-radius:50%; vertical-align:-1px; }
    .muted { color:#aab2bd; }
  </style>
  <script type="importmap">{"imports":{"three":"../../../viewer/vendor/three/build/three.module.js","three/addons/":"../../../viewer/vendor/three/examples/jsm/"}}</script>
</head>
<body>
  <div id="hud">
    <strong>Incremental no-LiDAR reconstruction</strong>
    <div class="row"><span id="status">loading</span><span id="frameLabel"></span></div>
    <label>Cumulative <select id="stageSelect"></select></label>
    <div id="layers"></div>
    <input id="frame" type="range" min="0" max="0" value="0" step="1" />
    <div id="closest" class="muted"></div>
    <div id="legend"></div>
  </div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

    const stageOptions = __STAGE_OPTIONS__;
    const layerOptions = __LAYER_OPTIONS__;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07090c);
    const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.01, 2500);
    camera.position.set(0, 18, 45);
    const renderer = new THREE.WebGLRenderer({ antialias:true });
    renderer.setSize(innerWidth, innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    document.body.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.GridHelper(140, 70, 0x38424c, 0x20262d));
    scene.add(new THREE.AxesHelper(4));
    scene.add(new THREE.HemisphereLight(0xffffff, 0x1b2430, 1.15));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.45);
    keyLight.position.set(18, 36, 24);
    scene.add(keyLight);
    const cloudGroup = new THREE.Group();
    const dynamicGroup = new THREE.Group();
    scene.add(cloudGroup);
    scene.add(dynamicGroup);

    const statusEl = document.getElementById('status');
    const stageSelectEl = document.getElementById('stageSelect');
    const layersEl = document.getElementById('layers');
    const frameEl = document.getElementById('frame');
    const frameLabelEl = document.getElementById('frameLabel');
    const closestEl = document.getElementById('closest');
    const legendEl = document.getElementById('legend');
    const loader = new GLTFLoader();
    const cache = new Map();
    const activeLayers = new Map();
    let stageObject = null;
    let fittedCamera = false;
    let tracks = null;
    let frames = [];
    const markers = [];

    function setStatus(message) {
      statusEl.textContent = message;
    }

    function loadCloud(key, url) {
      if (cache.has(key)) return cache.get(key);
      const promise = new Promise((resolve, reject) => {
        loader.load(url, gltf => resolve(gltf.scene), undefined, reject);
      });
      cache.set(key, promise);
      return promise;
    }

    function fitCameraToObject(object) {
      const box = new THREE.Box3().setFromObject(object);
      if (box.isEmpty()) return;
      const center = box.getCenter(new THREE.Vector3());
      const size = Math.max(10, box.getSize(new THREE.Vector3()).length());
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(size * 0.25, size * 0.18, size * 0.42));
      camera.near = Math.max(0.01, size / 10000);
      camera.far = Math.max(2500, size * 10);
      camera.updateProjectionMatrix();
      fittedCamera = true;
    }

    async function setStage(option) {
      setStatus(`loading ${option.id}`);
      const object = await loadCloud(`stage:${option.id}`, option.url);
      if (stageObject) cloudGroup.remove(stageObject);
      stageObject = object;
      cloudGroup.add(stageObject);
      if (!fittedCamera) fitCameraToObject(stageObject);
      setStatus(`${option.id}: ${option.points.toLocaleString()} pts`);
    }

    async function toggleLayer(option, enabled) {
      if (!enabled) {
        const object = activeLayers.get(option.id);
        if (object) cloudGroup.remove(object);
        activeLayers.delete(option.id);
        return;
      }
      setStatus(`loading ${option.id}`);
      const object = await loadCloud(`layer:${option.id}`, option.url);
      activeLayers.set(option.id, object);
      cloudGroup.add(object);
      setStatus(`${option.id}: +${option.points.toLocaleString()} pts`);
    }

    stageOptions.forEach((option, index) => {
      const node = document.createElement('option');
      node.value = String(index);
      node.textContent = option.label;
      stageSelectEl.appendChild(node);
    });
    stageSelectEl.value = String(Math.max(0, stageOptions.length - 1));
    stageSelectEl.addEventListener('change', () => setStage(stageOptions[Number(stageSelectEl.value)]));

    layerOptions.forEach(option => {
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.addEventListener('change', () => toggleLayer(option, checkbox.checked));
      label.appendChild(checkbox);
      label.appendChild(document.createTextNode(option.label));
      layersEl.appendChild(label);
    });

    function hexToNumber(hex) {
      return Number.parseInt(hex.replace('#', ''), 16);
    }

    function makeLine(points, color) {
      const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
      const material = new THREE.LineBasicMaterial({ color });
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
        dynamicGroup.add(makeLine(data.samples.map(s => s.position_viewer), color));
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
    setStage(stageOptions[Math.max(0, stageOptions.length - 1)]).catch(err => {
      console.error(err);
      setStatus('stage failed');
    });

    Promise.all([
      fetch('../replay/agent_tracks.json').then(r => r.json()),
      fetch('../replay/accident_diagnostics.json').then(r => r.json())
    ]).then(([trackJson, diagJson]) => {
      tracks = trackJson;
      addTracks();
      const c = diagJson.closest_approach;
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
"""
    html = html.replace("__STAGE_OPTIONS__", json.dumps(stages))
    html = html.replace("__LAYER_OPTIONS__", json.dumps(layers))
    (viewer / "index.html").write_text(html, encoding="utf-8")


def cap_points(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[idx], colors[idx]


def main() -> int:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.source = args.source.resolve()
    args.out = args.out.resolve()
    args.mask_export = args.mask_export.resolve()
    args.slam3r_root = args.slam3r_root.resolve()
    base.CATEGORY = args.category

    clean_output(args.out)
    start = time.time()
    base_points, base_colors = read_ply_points(args.source / "reconstruction" / "points_world.ply")
    source_alignment = json.loads((args.source / "reconstruction" / "alignment.json").read_text(encoding="utf-8"))
    origin = np.asarray(source_alignment["viewer_origin_world"], dtype=np.float64)
    streams = make_streams(args)

    cumulative_points = base_points
    cumulative_colors = base_colors
    occupied = occupied_keys(cumulative_points, args.add_voxel)
    stage_reports = []

    base_stage = write_cloud_bundle(
        args.out / "stages" / "00_base",
        cumulative_points,
        cumulative_colors,
        origin,
        "00 base no-LiDAR SLAM3R calibrated reconstruction",
        args.seed,
    )
    stage_reports.append(
        {
            "stage": "00_base",
            "description": "Current good result used as immutable base.",
            "new_points": int(len(base_points)),
            "cumulative_points": int(len(cumulative_points)),
            "bundle": base_stage,
        }
    )

    layer_reports = []
    for layer_index, spec in enumerate(layer_specs(), start=1):
        fuse_args = make_fuse_args(args, spec)
        points, colors, alignment = slam3r.fuse_streams(streams, fuse_args)
        points, colors = slam3r.voxel_downsample_world(points, colors, args.voxel)
        points, colors = cap_points(points, colors, args.max_points, args.seed + layer_index)
        new_points, new_colors, duplicate_count = select_new_voxels(points, colors, occupied, args.add_voxel)

        layer_dir = args.out / "layers" / spec.name
        layer_bundle = write_cloud_bundle(
            layer_dir,
            new_points,
            new_colors,
            origin,
            f"{spec.name} additions only",
            args.seed + layer_index,
        )
        base.write_json(layer_dir / "alignment.json", alignment)

        if len(new_points):
            cumulative_points = np.vstack([cumulative_points, new_points])
            cumulative_colors = np.vstack([cumulative_colors, new_colors])
            cumulative_points, cumulative_colors = slam3r.voxel_downsample_world(cumulative_points, cumulative_colors, args.voxel)
            cumulative_points, cumulative_colors = cap_points(
                cumulative_points,
                cumulative_colors,
                args.max_points,
                args.seed + layer_index + 100,
            )
            occupied = occupied_keys(cumulative_points, args.add_voxel)

        stage_name = f"{layer_index:02d}_{spec.name}"
        stage_bundle = write_cloud_bundle(
            args.out / "stages" / stage_name,
            cumulative_points,
            cumulative_colors,
            origin,
            f"{stage_name} cumulative no-LiDAR reconstruction",
            args.seed + layer_index,
        )
        layer_report = {
            "layer": spec.name,
            "spec": asdict(spec),
            "candidate_points": int(len(points)),
            "duplicate_voxel_points": int(duplicate_count),
            "new_points": int(len(new_points)),
            "cumulative_points": int(len(cumulative_points)),
            "accepted_stream_count": int(alignment["accepted_stream_count"]),
            "rejected_stream_count": int(alignment["rejected_stream_count"]),
            "alignment_residual": alignment["aggregate_residual"],
            "bundle": layer_bundle,
        }
        stage_report = {
            "stage": stage_name,
            "description": spec.description,
            "new_points": int(len(new_points)),
            "cumulative_points": int(len(cumulative_points)),
            "bundle": stage_bundle,
        }
        layer_reports.append(layer_report)
        stage_reports.append(stage_report)
        base.write_json(layer_dir / "summary.json", layer_report)
        base.write_json(args.out / "stages" / stage_name / "summary.json", stage_report)

    final_viewer_points, _ = base.world_to_viewer(cumulative_points, origin=origin)
    base.write_ply(args.out / "reconstruction" / "points_world.ply", cumulative_points, cumulative_colors)
    base.write_ply(args.out / "reconstruction" / "points.ply", final_viewer_points, cumulative_colors)
    base.export_scene_glb(args.out / "scene.glb", final_viewer_points, cumulative_colors)
    base.write_preview(args.out / "reports" / "preview_point_cloud.png", final_viewer_points, cumulative_colors)
    write_topdown(args.out / "reports" / "topdown_preview.png", cumulative_points, cumulative_colors, "Final cumulative no-LiDAR layers", args.seed)

    frames = sorted({frame for stream in streams for frame in stream.frames})
    tracks = multi.build_agent_tracks(args.dataset, args.out, args.agents, frames, args.scenario, origin, args.nominal_fps)
    object_context = multi.build_object_tracks(args.dataset, args.out, args.agents, frames, args.scenario)
    diagnostics = multi.accident_diagnostics(args.out, tracks)
    multi.write_viewer(args.out)
    write_incremental_viewer(args.out, stage_reports, layer_reports)

    summary = {
        "schema_version": "0.1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "ok",
        "backend": "slam3r_incremental_camera_ray_depth_layers",
        "source": str(args.source),
        "scenario": args.scenario,
        "lidar_used": False,
        "legacy_lidar_assets_used": False,
        "calibration_used": True,
        "rgb_used": True,
        "stage_count": len(stage_reports),
        "layer_count": len(layer_reports),
        "base_point_count": int(len(base_points)),
        "final_point_count": int(len(cumulative_points)),
        "world_origin": origin.tolist(),
        "bbox_min_world": [float(x) for x in cumulative_points.min(axis=0)],
        "bbox_max_world": [float(x) for x in cumulative_points.max(axis=0)],
        "object_context_records": int(object_context["record_count"]),
        "elapsed_seconds": time.time() - start,
        "outputs": {
            "scene_glb": "scene.glb",
            "points_world_ply": "reconstruction/points_world.ply",
            "preview": "reports/preview_point_cloud.png",
            "topdown_preview": "reports/topdown_preview.png",
            "viewer": "viewer/index.html",
            "stages": "stages/",
            "layers": "layers/",
        },
        "stages": stage_reports,
        "layers": layer_reports,
        "diagnostics": {
            "closest_approach": diagnostics.get("closest_approach"),
        },
    }
    manifest = {
        "schema_version": "0.1",
        "case_id": "town04_type1_subtype2_slam3r_incremental_layers",
        "status": summary["status"],
        "created_utc": summary["created_utc"],
        "inputs": {
            "source": str(args.source),
            "dataset_root": str(args.dataset),
            "mask_export": str(args.mask_export),
            "slam3r_root": str(args.slam3r_root),
            "scenario": args.scenario,
            "agents": args.agents,
            "cameras": args.cameras,
        },
        "outputs": summary["outputs"],
        "quality_summary": {
            "backend": summary["backend"],
            "base_point_count": summary["base_point_count"],
            "final_point_count": summary["final_point_count"],
            "added_point_count": int(summary["final_point_count"] - summary["base_point_count"]),
            "stage_count": summary["stage_count"],
        },
        "lidar_used": False,
        "legacy_lidar_assets_used": False,
    }
    base.write_json(args.out / "summary.json", summary)
    base.write_json(args.out / "manifest.json", manifest)
    report = f"""# Incremental No-LiDAR SLAM3R Layers

Status: `{summary['status']}`

Base: `{args.source}`

Method:

- Base layer is the current good no-LiDAR calibrated SLAM3R reconstruction.
- Additional layers reuse saved SLAM3R RGB point maps and DeepAccident camera calibration.
- New points are accepted only when their `{args.add_voxel}` m voxel is not already occupied by the cumulative result.
- The default run has no point cap, so the base layer remains preserved in the cumulative result.
- `lidar01` point clouds are not read.

Final point count: `{summary['final_point_count']}`
Added point count: `{summary['final_point_count'] - summary['base_point_count']}`

Stages:

""" + "\n".join(
        f"- `{item['stage']}`: +`{item['new_points']}` points, cumulative `{item['cumulative_points']}`"
        for item in stage_reports
    )
    (args.out / "reports" / "incremental_layers_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
