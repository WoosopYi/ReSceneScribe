#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('RESCENESCRIBE_OUTPUT_ROOT', REPO_ROOT)).expanduser().resolve()
DATASET = Path(os.environ.get('DEEPACCIDENT_ROOT', REPO_ROOT / 'deepaccident_mini_dataset')).expanduser().resolve()
_SOURCE_VIEWER_ENV = os.environ.get('RESCENESCRIBE_SOURCE_VIEWER')
SOURCE_VIEWER = Path(_SOURCE_VIEWER_ENV).expanduser() if _SOURCE_VIEWER_ENV else None
CATEGORY = os.environ.get('DEEPACCIDENT_CATEGORY', 'type1_subtype1_accident')
SCENARIO = os.environ.get('DEEPACCIDENT_SCENARIO', 'Town03_type001_subtype0001_scenario00024')
OUT = ROOT / 'viewer'
ASSETS = ROOT / 'viewer_assets'
FRAMES = ROOT / 'viewer_frames'
PREVIEWS = ROOT / 'previews'

CAMERA_NAMES = [
    'Camera_Front',
    'Camera_FrontLeft',
    'Camera_FrontRight',
    'Camera_BackLeft',
    'Camera_BackRight',
    'Camera_Back',
]

AGENTS = [
    {
        'key': 'ego_accident',
        'agent': 'ego_vehicle',
        'name': 'Collision vehicle A',
        'role': 'colliding ego vehicle',
        'camera': 'Camera_Front',
        'color': '#ff6536',
        'lane_group': 'A',
        'order': 0,
    },
    {
        'key': 'ego_follower',
        'agent': 'ego_vehicle_behind',
        'name': 'Follower behind A',
        'role': 'vehicle following collision vehicle A',
        'camera': 'Camera_Front',
        'color': '#28d7a8',
        'lane_group': 'A',
        'order': 1,
    },
    {
        'key': 'other_accident',
        'agent': 'other_vehicle',
        'name': 'Collision vehicle B',
        'role': 'colliding other vehicle',
        'camera': 'Camera_Front',
        'color': '#4aa3ff',
        'lane_group': 'B',
        'order': 2,
    },
    {
        'key': 'other_follower',
        'agent': 'other_vehicle_behind',
        'name': 'Follower behind B',
        'role': 'vehicle following collision vehicle B',
        'camera': 'Camera_Front',
        'color': '#d29cff',
        'lane_group': 'B',
        'order': 3,
    },
]


def carla_to_viewer(vec: np.ndarray) -> np.ndarray:
    return np.asarray([vec[0], vec[2], vec[1]], dtype=np.float64)


def scenario_path(agent: str, sensor: str) -> Path:
    return DATASET / CATEGORY / agent / sensor / SCENARIO


def load_calib(agent: str, frame: int) -> dict:
    path = scenario_path(agent, 'calib') / f'{SCENARIO}_{frame:03d}.pkl'
    with path.open('rb') as f:
        return pickle.load(f)


def frame_count(agent: str, camera: str = 'Camera_Front') -> int:
    return len(sorted(scenario_path(agent, camera).glob('*.jpg')))


def parse_meta_info() -> dict:
    path = DATASET / CATEGORY / 'meta' / f'{SCENARIO}.txt'
    lines = path.read_text(errors='replace').splitlines()
    first = lines[0].split() if lines else []
    info = {
        'path': str(path),
        'raw_first_line': ' '.join(first),
        'weather': first[0] if first else None,
        'metadata_accident_frame': int(first[-1]) if first and first[-1].lstrip('-').isdigit() else None,
        'colliding_agents': [],
        'agent_ids': [],
        'road_type': None,
    }
    for line in lines[1:]:
        if 'colliding agents:' in line:
            info['colliding_agents'] = line.split(':', 1)[1].split()
        elif 'agents id:' in line:
            info['agent_ids'] = line.split(':', 1)[1].split()
        elif 'road_type:' in line:
            info['road_type'] = line.split(':', 1)[1].strip()
    return info


def sat_gap_from_matrices(a: dict, b: dict) -> tuple[float, float, bool]:
    ma = np.asarray(a['vehicle_matrix'], dtype=np.float64)
    mb = np.asarray(b['vehicle_matrix'], dtype=np.float64)
    da = a['dimensions_m']
    db = b['dimensions_m']
    ca = ma[[0, 2], 3]
    cb = mb[[0, 2], 3]
    axes = []
    for m in [ma, mb]:
        for col in [0, 2]:
            axis = m[[0, 2], col]
            axis = axis / max(np.linalg.norm(axis), 1e-9)
            axes.append(axis)
    half_a = [da['length'] / 2.0, da['width'] / 2.0]
    half_b = [db['length'] / 2.0, db['width'] / 2.0]
    axes_a = [ma[[0, 2], 0] / max(np.linalg.norm(ma[[0, 2], 0]), 1e-9), ma[[0, 2], 2] / max(np.linalg.norm(ma[[0, 2], 2]), 1e-9)]
    axes_b = [mb[[0, 2], 0] / max(np.linalg.norm(mb[[0, 2], 0]), 1e-9), mb[[0, 2], 2] / max(np.linalg.norm(mb[[0, 2], 2]), 1e-9)]
    max_gap = -1e9
    overlap = True
    for axis in axes:
        c1 = float(np.dot(ca, axis))
        c2 = float(np.dot(cb, axis))
        r1 = sum(abs(float(np.dot(ax, axis))) * h for ax, h in zip(axes_a, half_a))
        r2 = sum(abs(float(np.dot(ax, axis))) * h for ax, h in zip(axes_b, half_b))
        gap = abs(c2 - c1) - (r1 + r2)
        max_gap = max(max_gap, gap)
        if gap > 0:
            overlap = False
    return float(max_gap), float(np.linalg.norm(ca - cb)), overlap


def build_collision_diagnostics(sequences: list[dict], meta: dict) -> dict:
    role_to_key = {
        'ego': 'ego_accident',
        'ego_behind': 'ego_follower',
        'other': 'other_accident',
        'other_behind': 'other_follower',
    }
    seq_by_key = {seq['key']: seq for seq in sequences}
    colliding_roles = meta.get('colliding_agents') or []
    mapped_keys = [role_to_key.get(role) for role in colliding_roles]
    if len(mapped_keys) != 2 or not all(mapped_keys):
        return {'meta_colliding_agents': colliding_roles, 'valid': False}
    a = seq_by_key[mapped_keys[0]]
    b = seq_by_key[mapped_keys[1]]
    best = None
    overlaps = []
    for idx in range(min(a['frame_count'], b['frame_count'])):
        fa = {**a['frames'][idx], 'dimensions_m': a['dimensions_m']}
        fb = {**b['frames'][idx], 'dimensions_m': b['dimensions_m']}
        gap, center_distance, overlap = sat_gap_from_matrices(fa, fb)
        if best is None or gap < best['obb_gap_m']:
            best = {
                'frame_index': idx,
                'source_frame': idx + 1,
                'obb_gap_m': gap,
                'center_distance_m': center_distance,
                'overlap': overlap,
            }
        if overlap:
            overlaps.append(idx + 1)
    return {
        'valid': True,
        'meta_colliding_agents': colliding_roles,
        'mapped_vehicle_keys': mapped_keys,
        'mapped_vehicle_names': [a['name'], b['name']],
        'closest_or_overlap_frame': best,
        'overlap_source_frames': overlaps,
        'interpretation': 'negative obb_gap_m means the two selected DeepAccident agent boxes overlap in the shared metric world',
    }


def parse_self_label(agent: str, frame: int = 1) -> dict:
    label_path = scenario_path(agent, 'label') / f'{SCENARIO}_{frame:03d}.txt'
    for line in label_path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) < 11:
            continue
        if parts[10] == '-100':
            return {
                'type': parts[0],
                'length': float(parts[4]),
                'width': float(parts[5]),
                'height': float(parts[6]),
                'raw': line,
            }
    raise ValueError(f'No self (-100) label found in {label_path}')


def vehicle_matrix_from_calib(calib: dict, origin: np.ndarray) -> np.ndarray:
    ego_to_world = np.asarray(calib['ego_to_world'], dtype=np.float64)
    rotation = ego_to_world[:3, :3]
    translation = carla_to_viewer(ego_to_world[:3, 3]) - origin

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = carla_to_viewer(rotation[:, 0])  # local forward
    matrix[:3, 1] = carla_to_viewer(rotation[:, 2])  # local up
    matrix[:3, 2] = carla_to_viewer(rotation[:, 1])  # local right
    matrix[:3, 3] = translation
    return matrix


def camera_world_path(agent: str, frame_total: int) -> np.ndarray:
    positions = []
    for frame in range(1, frame_total + 1):
        calib = load_calib(agent, frame)
        camera_to_world = (
            np.asarray(calib['ego_to_world'], dtype=np.float64)
            @ np.asarray(calib['lidar_to_ego'], dtype=np.float64)
            @ np.linalg.inv(np.asarray(calib['lidar_to_Camera_Front'], dtype=np.float64))
        )
        positions.append(carla_to_viewer(camera_to_world[:3, 3]))
    return np.asarray(positions, dtype=np.float64)


def shared_origin(agent_frame_count: int) -> np.ndarray:
    paths = [camera_world_path(agent['agent'], agent_frame_count) for agent in AGENTS]
    all_positions = np.vstack(paths)
    return np.array(
        [
            (all_positions[:, 0].min() + all_positions[:, 0].max()) / 2.0,
            all_positions[:, 1].mean(),
            (all_positions[:, 2].min() + all_positions[:, 2].max()) / 2.0,
        ],
        dtype=np.float64,
    )


def build_vehicle_sequences(origin: np.ndarray, agent_frame_count: int) -> list[dict]:
    sequences = []
    for agent_spec in AGENTS:
        dims = parse_self_label(agent_spec['agent'], 1)
        frames = []
        previous = None
        cumulative = 0.0
        for frame in range(1, agent_frame_count + 1):
            calib = load_calib(agent_spec['agent'], frame)
            matrix = vehicle_matrix_from_calib(calib, origin)
            position = matrix[:3, 3]
            step = 0.0 if previous is None else float(np.linalg.norm(position - previous))
            cumulative += step
            previous = position
            frames.append(
                {
                    'index': frame - 1,
                    'source_frame': frame,
                    'vehicle_position': [float(v) for v in position],
                    'vehicle_matrix': matrix.tolist(),
                    'step_motion': step,
                    'cumulative_motion': cumulative,
                }
            )
        sequences.append(
            {
                **agent_spec,
                'vehicle_type': dims['type'],
                'dimensions_m': {
                    'length': dims['length'],
                    'width': dims['width'],
                    'height': dims['height'],
                },
                'frame_pattern': f'../viewer_frames/{agent_spec["agent"]}/frame_{{index4}}.jpg',
                'frames': frames,
                'frame_count': len(frames),
                'total_path_length_m': cumulative,
                'net_displacement_m': float(np.linalg.norm(np.asarray(frames[-1]['vehicle_position']) - np.asarray(frames[0]['vehicle_position']))),
            }
        )
    return sequences


def trajectory_bounds(sequences: list[dict], margin: float = 16.0) -> dict:
    pts = []
    for seq in sequences:
        for frame in seq['frames']:
            pts.append(frame['vehicle_position'])
    arr = np.asarray(pts, dtype=np.float64)
    mins = arr.min(axis=0) - np.array([margin, 2.0, margin])
    maxs = arr.max(axis=0) + np.array([margin, 8.0, margin])
    center = (mins + maxs) / 2.0
    span = maxs - mins
    return {
        'min': [float(v) for v in mins],
        'max': [float(v) for v in maxs],
        'center': [float(v) for v in center],
        'span': [float(v) for v in span],
    }


def copy_dashcam_frames(frame_total: int) -> None:
    for agent_spec in AGENTS:
        target = FRAMES / agent_spec['agent']
        target.mkdir(parents=True, exist_ok=True)
        source = scenario_path(agent_spec['agent'], agent_spec['camera'])
        for i in range(1, frame_total + 1):
            src = source / f'{SCENARIO}_{i:03d}.jpg'
            dst = target / f'frame_{i - 1:04d}.jpg'
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                # Browser display does not need original 1600px frames; resize to keep the viewer light.
                im = Image.open(src).convert('RGB')
                im.thumbnail((960, 540), Image.Resampling.LANCZOS)
                im.save(dst, quality=86, optimize=True)


def project_lidar_colors(agent: str, frame: int, calib: dict, lidar_points: np.ndarray) -> np.ndarray:
    intensity = lidar_points[:, 3] if lidar_points.shape[1] > 3 else np.zeros(len(lidar_points), dtype=np.float64)
    lo, hi = np.percentile(intensity, 2), np.percentile(intensity, 98)
    norm = np.clip((intensity - lo) / max(hi - lo, 1e-6), 0, 1)
    colors = np.repeat((65 + norm[:, None] * 145).astype(np.uint8), 3, axis=1)
    assigned = np.zeros(len(lidar_points), dtype=bool)
    homogeneous = np.concatenate([lidar_points[:, :3], np.ones((len(lidar_points), 1), dtype=np.float64)], axis=1)

    for camera in CAMERA_NAMES:
        image_path = scenario_path(agent, camera) / f'{SCENARIO}_{frame:03d}.jpg'
        if not image_path.exists():
            continue
        image = np.asarray(Image.open(image_path).convert('RGB'))
        height, width = image.shape[:2]
        lidar_to_camera = np.asarray(calib[f'lidar_to_{camera}'], dtype=np.float64)
        intrinsic = np.asarray(calib[f'intrinsic_{camera}'], dtype=np.float64)
        cam_points = (lidar_to_camera @ homogeneous.T).T[:, :3]
        projected = (intrinsic @ cam_points.T).T
        depth = projected[:, 2]
        valid_depth = depth > 1e-3
        u = projected[:, 0] / np.where(valid_depth, depth, 1.0)
        v = projected[:, 1] / np.where(valid_depth, depth, 1.0)
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        valid = valid_depth & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height) & ~assigned
        if np.any(valid):
            colors[valid] = image[vi[valid], ui[valid]]
            assigned[valid] = True
        if assigned.all():
            break
    return colors


def points_inside_vehicle_box(points: np.ndarray, matrix: np.ndarray, dims: dict, padding: float = 0.45) -> np.ndarray:
    center = matrix[:3, 3]
    rotation = matrix[:3, :3]
    local = (points - center) @ rotation
    return (
        (np.abs(local[:, 0]) <= dims['length'] * 0.5 + padding)
        & (local[:, 1] >= 0.12)
        & (local[:, 1] <= dims['height'] + padding)
        & (np.abs(local[:, 2]) <= dims['width'] * 0.5 + padding)
    )


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int32)
    _, keep_idx = np.unique(keys, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    points = points[keep_idx]
    colors = colors[keep_idx]
    if len(points) > limit:
        rng = np.random.default_rng(seed)
        keep_idx = np.sort(rng.choice(len(points), size=limit, replace=False))
        points = points[keep_idx]
        colors = colors[keep_idx]
    return points, colors


def build_static_lidar_scene(data: dict, frame_total: int) -> dict:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out_path = ASSETS / 'four_vehicle_static_lidar_background.glb'
    ultra_path = ASSETS / 'four_vehicle_static_lidar_background_ultra.glb'
    origin = np.asarray(data['metric_alignment']['origin_viewer_xyz'], dtype=np.float64)
    bounds = data['scene_bounds']

    # Precompute four vehicle boxes for each frame so dynamic vehicle returns are removed from the static background.
    box_by_frame: dict[int, list[tuple[np.ndarray, dict]]] = {}
    for frame_idx in range(frame_total):
        boxes = []
        for seq in data['sequences']:
            boxes.append((np.asarray(seq['frames'][frame_idx]['vehicle_matrix'], dtype=np.float64), seq['dimensions_m']))
        box_by_frame[frame_idx + 1] = boxes

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    source_counts: dict[str, int] = {}
    kept_counts: dict[str, int] = {}
    dynamic_removed_counts: dict[str, int] = {}

    for agent_spec in AGENTS:
        agent = agent_spec['agent']
        source_counts[agent] = 0
        kept_counts[agent] = 0
        dynamic_removed_counts[agent] = 0
        lidar_folder = scenario_path(agent, 'lidar01')
        for frame in range(1, frame_total + 1):
            lidar_path = lidar_folder / f'{SCENARIO}_{frame:03d}.npz'
            if not lidar_path.exists():
                continue
            calib = load_calib(agent, frame)
            lidar_points = np.load(lidar_path)['data'].astype(np.float64)
            source_counts[agent] += int(len(lidar_points))

            homogeneous = np.concatenate([lidar_points[:, :3], np.ones((len(lidar_points), 1), dtype=np.float64)], axis=1)
            world = (
                np.asarray(calib['ego_to_world'], dtype=np.float64)
                @ np.asarray(calib['lidar_to_ego'], dtype=np.float64)
                @ homogeneous.T
            ).T[:, :3]
            viewer_points = np.column_stack([world[:, 0], world[:, 2], world[:, 1]]) - origin
            colors = project_lidar_colors(agent, frame, calib, lidar_points)

            crop = (
                (viewer_points[:, 0] >= bounds['min'][0])
                & (viewer_points[:, 0] <= bounds['max'][0])
                & (viewer_points[:, 1] >= bounds['min'][1])
                & (viewer_points[:, 1] <= bounds['max'][1])
                & (viewer_points[:, 2] >= bounds['min'][2])
                & (viewer_points[:, 2] <= bounds['max'][2])
            )
            if not np.any(crop):
                continue
            points = viewer_points[crop]
            colors = colors[crop]
            dynamic = np.zeros(len(points), dtype=bool)
            for matrix, dims in box_by_frame[frame]:
                dynamic |= points_inside_vehicle_box(points, matrix, dims)
            dynamic_removed_counts[agent] += int(dynamic.sum())
            keep = ~dynamic
            kept_counts[agent] += int(keep.sum())
            if np.any(keep):
                all_points.append(points[keep].astype(np.float32))
                all_colors.append(colors[keep].astype(np.uint8))

    if not all_points:
        raise RuntimeError('No lidar points survived crop/filter')

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    raw_kept = int(len(points))

    ultra_points, ultra_colors = voxel_downsample(points, colors, voxel_size=0.040, limit=2_200_000, seed=4004)
    trimesh.PointCloud(vertices=ultra_points, colors=ultra_colors).export(ultra_path)

    regular_points, regular_colors = voxel_downsample(points, colors, voxel_size=0.075, limit=1_050_000, seed=4005)
    trimesh.PointCloud(vertices=regular_points, colors=regular_colors).export(out_path)

    return {
        'glb': '../viewer_assets/four_vehicle_static_lidar_background.glb',
        'ultra_glb': '../viewer_assets/four_vehicle_static_lidar_background_ultra.glb',
        'point_count': int(len(regular_points)),
        'ultra_point_count': int(len(ultra_points)),
        'raw_static_points_before_voxel': raw_kept,
        'source_point_counts': source_counts,
        'kept_static_point_counts': kept_counts,
        'removed_dynamic_vehicle_point_counts': dynamic_removed_counts,
        'filter': 'Four-agent DeepAccident LiDAR fused into one metric world; points inside the four moving vehicle boxes are removed so the background stays static while the vehicles move separately.',
    }


def make_topdown_preview(data: dict) -> None:
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    size = 1400
    margin = 120
    img = Image.new('RGB', (size, size), (10, 11, 13))
    draw = ImageDraw.Draw(img)
    pts = []
    for seq in data['sequences']:
        for frame in seq['frames']:
            p = frame['vehicle_position']
            pts.append((p[0], p[2]))
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    span = max(max(xs) - min(xs), max(zs) - min(zs), 1e-6)
    scale = (size - margin * 2) / span
    cx = (max(xs) + min(xs)) / 2.0
    cz = (max(zs) + min(zs)) / 2.0

    def project(x: float, z: float) -> tuple[float, float]:
        return (size / 2 + (x - cx) * scale, size / 2 - (z - cz) * scale)

    for g in range(-8, 9):
        c = (30, 33, 37)
        x = size / 2 + g * 70
        y = size / 2 + g * 70
        draw.line((x, margin * 0.5, x, size - margin * 0.5), fill=c, width=1)
        draw.line((margin * 0.5, y, size - margin * 0.5, y), fill=c, width=1)

    for seq in data['sequences']:
        path = [project(f['vehicle_position'][0], f['vehicle_position'][2]) for f in seq['frames']]
        draw.line(path, fill=seq['color'], width=7)
        for idx, pt in enumerate(path):
            if idx in (0, len(path) - 1) or idx % 8 == 0:
                r = 9 if idx == len(path) - 1 else 5
                draw.ellipse((pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r), fill=seq['color'])
        label_pt = path[-1]
        draw.text((label_pt[0] + 10, label_pt[1] - 12), seq['name'], fill=seq['color'])
    draw.text((32, 28), f'DeepAccident {SCENARIO} four-vehicle reconstructed motion, frames 001-{len(data["sequences"][0]["frames"]):03d}', fill=(238, 238, 232))
    draw.text((32, 58), 'Static background = four-agent LiDAR fusion with moving-vehicle boxes removed; cars = calibration-derived body poses.', fill=(175, 178, 174))
    img.save(PREVIEWS / 'four_vehicle_topdown_plan.png')


def write_scene_data(data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'four_vehicle_scene_data.js').write_text(
        'window.FOUR_VEHICLE_SCENE_DATA = ' + json.dumps(data, separators=(',', ':')) + ';\n'
    )


def copy_three_vendor() -> None:
    vendor = OUT / 'vendor'
    if (vendor / 'three/build/three.module.js').exists():
        return

    candidates = []
    if SOURCE_VIEWER is not None:
        candidates.append(SOURCE_VIEWER / 'vendor')
    candidates.append(REPO_ROOT / 'viewer/vendor')
    for src_vendor in candidates:
        if not src_vendor.exists():
            continue
        if vendor.exists():
            shutil.rmtree(vendor)
        shutil.copytree(src_vendor, vendor)
        return

    # Optional fallback for environments that keep Three.js outside this repo.
    three_src = Path(os.environ.get('THREE_ROOT', '')).expanduser()
    if not three_src.exists():
        raise FileNotFoundError(
            'Three.js vendor files were not found. Keep viewer/vendor in the repo, '
            'set RESCENESCRIBE_SOURCE_VIEWER, set THREE_ROOT, or install/copy three.js into the output viewer/vendor directory.'
        )
    (vendor / 'three/build').mkdir(parents=True, exist_ok=True)
    (vendor / 'three/examples/jsm/controls').mkdir(parents=True, exist_ok=True)
    (vendor / 'three/examples/jsm/loaders').mkdir(parents=True, exist_ok=True)
    (vendor / 'three/examples/jsm/utils').mkdir(parents=True, exist_ok=True)
    shutil.copy2(three_src / 'build/three.module.js', vendor / 'three/build/three.module.js')
    shutil.copy2(three_src / 'examples/jsm/controls/OrbitControls.js', vendor / 'three/examples/jsm/controls/OrbitControls.js')
    shutil.copy2(three_src / 'examples/jsm/loaders/GLTFLoader.js', vendor / 'three/examples/jsm/loaders/GLTFLoader.js')
    shutil.copy2(three_src / 'examples/jsm/utils/BufferGeometryUtils.js', vendor / 'three/examples/jsm/utils/BufferGeometryUtils.js')


def write_html() -> None:
    html = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ReSceneScribe Four Vehicle Collision Replay</title>
  <style>
    :root { color-scheme: dark; --bg:#090a0c; --panel:rgba(16,17,20,.91); --line:rgba(242,238,226,.16); --text:#f3f1ea; --muted:#b9b4aa; --accent:#ffc65a; }
    *{box-sizing:border-box} html,body{width:100%;height:100%;margin:0;overflow:hidden;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    #scene{position:fixed;inset:0;width:100%;height:100%;display:block}.panel{position:fixed;z-index:5;background:var(--panel);border:1px solid var(--line);border-radius:9px;backdrop-filter:blur(12px);box-shadow:0 18px 42px rgba(0,0,0,.38)}
    .hud{top:16px;left:16px;width:min(650px,calc(100vw - 32px));padding:14px 16px}.controls{right:16px;top:16px;width:min(470px,calc(100vw - 32px));padding:12px}.cams{left:16px;right:16px;bottom:16px;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;padding:10px;background:rgba(24,25,28,.94)}
    h1{margin:0 0 9px;font-size:18px;line-height:1.22}.meta{display:grid;grid-template-columns:142px minmax(0,1fr);gap:5px 12px;color:var(--muted);font-size:12px;line-height:1.35}.meta strong{color:var(--text);font-weight:680;overflow-wrap:anywhere}.legend{display:flex;flex-wrap:wrap;gap:7px 12px;margin-top:11px;font-size:12px;color:var(--muted)}.legend span{display:inline-flex;align-items:center;gap:6px}.swatch{width:10px;height:10px;border-radius:50%}
    .row{display:flex;align-items:center;gap:9px}button{appearance:none;height:34px;min-width:62px;border:1px solid var(--line);border-radius:6px;background:rgba(255,255,255,.08);color:var(--text);padding:0 12px;font:inherit;font-size:12px;cursor:pointer}button:hover{background:rgba(255,255,255,.14)}input[type=range]{width:100%;accent-color:var(--accent)}.frameLabel{min-width:88px;text-align:right;color:var(--muted);font-size:12px}.stats{margin-top:9px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;font-size:12px;color:var(--muted)}.stat{padding:8px;border:1px solid var(--line);border-radius:6px;background:rgba(255,255,255,.045);line-height:1.35;min-width:0}.stat b{font-weight:720}.cam{min-width:0;overflow:hidden;margin:0;border-radius:6px;border:1px solid var(--line);background:#111315}.cam img{display:block;width:100%;height:clamp(95px,14vh,170px);object-fit:cover}.cam figcaption{display:flex;justify-content:space-between;gap:8px;padding:7px 8px;color:var(--muted);font-size:11px;white-space:nowrap}.cam strong{color:var(--text);min-width:0;overflow:hidden;text-overflow:ellipsis}.loading{position:fixed;z-index:4;left:50%;top:50%;transform:translate(-50%,-50%);color:rgba(243,241,234,.75);font-size:13px;pointer-events:none}.badge{font-size:10px;border:1px solid var(--line);border-radius:99px;padding:1px 6px;margin-left:5px;color:var(--muted)}
    @media(max-width:1100px){.cams{grid-template-columns:repeat(2,minmax(0,1fr))}.controls{top:210px;left:16px;right:16px;width:auto}.hud{width:calc(100vw - 32px)}}
    @media(max-width:640px){.cams{grid-template-columns:1fr}.cam img{height:85px}.meta{grid-template-columns:105px minmax(0,1fr);font-size:11px}.stats{grid-template-columns:1fr}.controls{top:235px}}
  </style>
</head>
<body>
  <canvas id="scene"></canvas>
  <div class="loading" id="loading">loading four-vehicle reconstruction</div>
  <section class="panel hud"><h1>ReSceneScribe Four Vehicle Collision Replay</h1><div class="meta" id="meta"></div><div class="legend" id="legend"></div></section>
  <section class="panel controls"><div class="row"><button id="play" type="button">Play</button><button id="reset" type="button">Reset</button><input id="frame" type="range" min="0" value="0" step="1"/><span class="frameLabel" id="frameLabel"></span></div><div class="stats" id="stats"></div></section>
  <section class="panel cams" id="cams"></section>
  <script src="four_vehicle_scene_data.js"></script>
  <script type="importmap">{"imports":{"three":"./vendor/three/build/three.module.js","three/addons/":"./vendor/three/examples/jsm/"}}</script>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

    const data = window.FOUR_VEHICLE_SCENE_DATA;
    const params = new URLSearchParams(window.location.search);
    const highQuality = params.get('quality') === 'high' || params.get('quality') === 'ultra';
    const ultraQuality = params.get('quality') === 'ultra';
    const showTrailGhosts = params.has('trail');
    const maxFrame = data.source.viewer_frame_count - 1;
    const accidentIndex = data.source.viewer_accident_index;
    window.__FOUR_VEHICLE_READY = false;

    const canvas = document.getElementById('scene');
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: highQuality, powerPreference: 'high-performance', preserveDrawingBuffer: params.has('capture') });
    } catch (e) {
      document.getElementById('loading').textContent = 'WebGL context failed: enable browser hardware acceleration';
      throw e;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, ultraQuality ? 1.9 : highQuality ? 1.5 : 1.0));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x090a0c, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x090a0c, 105, 230);
    const bounds = data.trajectory_bounds;
    const center = new THREE.Vector3(...bounds.center);
    const radius = Math.max(...bounds.span, 24);
    const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.03, Math.max(420, radius * 7));
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.maxDistance = Math.max(260, radius * 5);
    controls.minDistance = 1.0;

    scene.add(new THREE.HemisphereLight(0xfff2d6, 0x22262b, 1.85));
    const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(4.5, 8.0, 4.0); scene.add(key);
    const fill = new THREE.DirectionalLight(0x8fd7ff, 0.75); fill.position.set(-6, 3, -4); scene.add(fill);

    const grid = new THREE.GridHelper(Math.max(140, radius * 2.4), 80, 0x5c5e64, 0x292b30);
    grid.position.y = data.metric_alignment.ground_y;
    grid.material.opacity = 0.34; grid.material.transparent = true; scene.add(grid);

    const loader = new GLTFLoader();
    const groups = new Map();
    const vehicles = new Map();
    const markers = new Map();

    function hexToNumber(hex){ return Number.parseInt(hex.replace('#',''),16); }
    function pathPosition(frame){ const p = frame.vehicle_position; return new THREE.Vector3(p[0], p[1] + 0.1, p[2]); }
    function applyPose(object, rows){ const m = rows; object.matrix.set(m[0][0],m[0][1],m[0][2],m[0][3],m[1][0],m[1][1],m[1][2],m[1][3],m[2][0],m[2][1],m[2][2],m[2][3],m[3][0],m[3][1],m[3][2],m[3][3]); object.matrixAutoUpdate = false; }

    function vehicleMaterials(seq, ghost=false){
      const color = hexToNumber(seq.color);
      return {
        body: new THREE.MeshStandardMaterial({color, transparent:true, opacity: ghost ? .14 : .96, roughness:.46, metalness:.22, emissive:color, emissiveIntensity: ghost ? .02 : .05}),
        glass: new THREE.MeshStandardMaterial({color:0xc6e5f4, transparent:true, opacity: ghost ? .05 : .58, roughness:.08, metalness:.05}),
        dark: new THREE.MeshStandardMaterial({color:0x101215, transparent:ghost, opacity: ghost ? .10 : 1, roughness:.75}),
        trim: new THREE.MeshStandardMaterial({color:0x20242a, transparent:ghost, opacity: ghost ? .10 : .9, roughness:.56, metalness:.2}),
        light: new THREE.MeshStandardMaterial({color:0xfff0bc, transparent:true, opacity: ghost ? .06 : .95, emissive:0xffd58d, emissiveIntensity:ghost?0:.8}),
        tail: new THREE.MeshStandardMaterial({color:0xb92223, transparent:true, opacity: ghost ? .06 : .92, emissive:0x8b1212, emissiveIntensity:ghost?0:.55})
      };
    }

    function makeCarLike(seq, ghost=false){
      const dims = seq.dimensions_m;
      const L=dims.length, W=dims.width, H=dims.height;
      const mats = vehicleMaterials(seq, ghost);
      const group = new THREE.Group();
      const bodyDepth = W * .92;
      const profile = new THREE.Shape();
      if (seq.vehicle_type === 'truck') {
        profile.moveTo(-L*.50,H*.18); profile.lineTo(L*.47,H*.18); profile.lineTo(L*.50,H*.68); profile.lineTo(L*.30,H*.72); profile.lineTo(L*.23,H*.94); profile.lineTo(-L*.35,H*.94); profile.lineTo(-L*.50,H*.70); profile.lineTo(-L*.50,H*.18);
      } else if (seq.vehicle_type === 'van') {
        profile.moveTo(-L*.50,H*.20); profile.lineTo(L*.47,H*.20); profile.quadraticCurveTo(L*.53,H*.35,L*.45,H*.55); profile.lineTo(L*.25,H*.86); profile.lineTo(-L*.42,H*.88); profile.quadraticCurveTo(-L*.53,H*.74,-L*.50,H*.20);
      } else {
        profile.moveTo(-L*.50,H*.28); profile.lineTo(-L*.40,H*.20); profile.lineTo(L*.32,H*.20); profile.quadraticCurveTo(L*.51,H*.24,L*.52,H*.38); profile.lineTo(L*.38,H*.50); profile.lineTo(L*.20,H*.60); profile.lineTo(L*.06,H*.84); profile.quadraticCurveTo(-L*.14,H*.94,-L*.30,H*.76); profile.lineTo(-L*.46,H*.50); profile.quadraticCurveTo(-L*.53,H*.40,-L*.50,H*.28);
      }
      const bodyGeo = new THREE.ExtrudeGeometry(profile, {depth:bodyDepth, bevelEnabled:true, bevelSize: ghost ? .025 : .055, bevelThickness: ghost ? .035 : .07, bevelSegments:ghost?1:3});
      bodyGeo.translate(0,0,-bodyDepth*.5); bodyGeo.computeVertexNormals();
      const body = new THREE.Mesh(bodyGeo,mats.body); group.add(body);
      const cabin = new THREE.Mesh(new THREE.BoxGeometry(L * (seq.vehicle_type === 'truck' ? .34 : .32), H*.24, W*.68), mats.glass);
      cabin.position.set(L * (seq.vehicle_type === 'truck' ? .18 : -.03), H*.72, 0); cabin.rotation.z = seq.vehicle_type==='truck' ? .08 : .22; group.add(cabin);
      const windshield = new THREE.Mesh(new THREE.BoxGeometry(L*.15, H*.22, W*.62), mats.glass); windshield.position.set(L*.23,H*.63,0); windshield.rotation.z=.45; group.add(windshield);
      const rearGlass = new THREE.Mesh(new THREE.BoxGeometry(L*.16,H*.20,W*.60),mats.glass); rearGlass.position.set(-L*.33,H*.58,0); rearGlass.rotation.z=-.42; group.add(rearGlass);
      const frontBumper = new THREE.Mesh(new THREE.BoxGeometry(.09,H*.16,W*.84),mats.trim); frontBumper.position.set(L*.515,H*.30,0); group.add(frontBumper);
      const rearBumper = new THREE.Mesh(new THREE.BoxGeometry(.09,H*.15,W*.84),mats.trim); rearBumper.position.set(-L*.505,H*.29,0); group.add(rearBumper);
      if (!ghost) {
        const wheelGeo = new THREE.CylinderGeometry(H*.18,H*.18,W*.13,28); wheelGeo.rotateX(Math.PI/2);
        const hubGeo = new THREE.CylinderGeometry(H*.085,H*.085,W*.145,20); hubGeo.rotateX(Math.PI/2);
        const hubMat = new THREE.MeshStandardMaterial({color:0xbac0c8,roughness:.35,metalness:.55});
        for (const x of [-L*.32,L*.32]) for (const z of [-W*.54,W*.54]) { const w = new THREE.Mesh(wheelGeo,mats.dark); w.position.set(x,H*.18,z); group.add(w); const h = new THREE.Mesh(hubGeo,hubMat); h.position.copy(w.position); group.add(h); }
        for (const z of [-W*.25,W*.25]) { const hl = new THREE.Mesh(new THREE.BoxGeometry(.04,H*.07,W*.18),mats.light); hl.position.set(L*.54,H*.40,z); group.add(hl); const tl = new THREE.Mesh(new THREE.BoxGeometry(.04,H*.075,W*.16),mats.tail); tl.position.set(-L*.535,H*.39,z); group.add(tl); }
        const label = makeLabelSprite(seq.name, seq.color); label.position.set(0,H+1.0,0); group.add(label);
      } else {
        const edge = new THREE.LineSegments(new THREE.EdgesGeometry(bodyGeo), new THREE.LineBasicMaterial({color:hexToNumber(seq.color), transparent:true, opacity:.34, depthTest:false, depthWrite:false})); group.add(edge);
      }
      group.traverse(n => { n.renderOrder = ghost ? 15 : 32; if (n.material) { const mats = Array.isArray(n.material)?n.material:[n.material]; for (const m of mats) { m.depthWrite=false; if(!ghost) m.depthTest=true; } } });
      return group;
    }

    function makeLabelSprite(text, color){
      const c = document.createElement('canvas'); c.width=512; c.height=96; const ctx=c.getContext('2d');
      ctx.fillStyle='rgba(0,0,0,.58)'; ctx.strokeStyle=color; ctx.lineWidth=4; ctx.roundRect(8,12,496,68,14); ctx.fill(); ctx.stroke();
      ctx.fillStyle='#fff'; ctx.font='bold 30px system-ui,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(text,256,47);
      const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
      const mat = new THREE.SpriteMaterial({map:tex, transparent:true, depthTest:false, depthWrite:false}); const sp = new THREE.Sprite(mat); sp.scale.set(7.5,1.4,1); sp.renderOrder=80; return sp;
    }

    function addTrajectory(seq, group){
      const color=hexToNumber(seq.color);
      const pts=seq.frames.map(f=>pathPosition(f));
      const fullCurve=new THREE.CatmullRomCurve3(pts);
      const fullPath=new THREE.Mesh(new THREE.TubeGeometry(fullCurve,Math.max(20,pts.length*5),.075,8,false),new THREE.MeshBasicMaterial({color,transparent:true,opacity:.25,depthTest:false,depthWrite:false})); fullPath.renderOrder=12; group.add(fullPath);
      const active = new THREE.Mesh(new THREE.TubeGeometry(fullCurve,Math.max(20,pts.length*5),.16,10,false),new THREE.MeshBasicMaterial({color,transparent:true,opacity:.86,depthTest:false,depthWrite:false})); active.renderOrder=14; group.add(active);
      const marker = new THREE.Mesh(new THREE.SphereGeometry(.32,24,14),new THREE.MeshBasicMaterial({color,transparent:true,opacity:.95,depthTest:false,depthWrite:false})); marker.renderOrder=38; group.add(marker);
      const start = new THREE.Mesh(new THREE.SphereGeometry(.11,20,10),new THREE.MeshBasicMaterial({color:0xf4f1e6,depthTest:false,depthWrite:false})); start.position.copy(pts[0]); start.renderOrder=16; group.add(start);
      markers.set(seq.key,{marker,active,points:pts});
    }

    function updateActivePath(seq, idx){
      const m=markers.get(seq.key); const pts=m.points.slice(0,idx+1); if(pts.length<2) pts.push(pts[0].clone().add(new THREE.Vector3(.01,0,0)));
      m.active.geometry.dispose(); m.active.geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), Math.max(8,pts.length*5), .16, 10, false);
    }

    function addGhosts(seq, group){ for(let i=0;i<seq.frames.length;i+=7){ const g=makeCarLike(seq,true); applyPose(g,seq.frames[i].vehicle_matrix); group.add(g); } }

    function tuneScene(object){ object.traverse(n=>{ if(n.isPoints && n.material){ n.frustumCulled=false; n.material.size = ultraQuality ? .035 : highQuality ? .045 : .060; n.material.sizeAttenuation=true; n.material.transparent=highQuality; n.material.opacity = highQuality ? .92 : 1; n.material.depthWrite=false; } if(n.isMesh){ n.frustumCulled=false; } }); }

    function fitCamera(){ controls.target.copy(center); camera.position.set(center.x + radius*.18, center.y + radius*.36 + 6, center.z + radius*.72 + 13); camera.updateProjectionMatrix(); controls.update(); }

    for (const seq of data.sequences) { const group = new THREE.Group(); groups.set(seq.key,group); scene.add(group); addTrajectory(seq,group); if(showTrailGhosts) addGhosts(seq,group); const v=makeCarLike(seq,false); applyPose(v,seq.frames[accidentIndex].vehicle_matrix); vehicles.set(seq.key,v); group.add(v); }

    const scenePath = ultraQuality && data.static_scene.ultra_glb ? data.static_scene.ultra_glb : data.static_scene.glb;
    loader.load(scenePath, gltf => { const model=gltf.scene; tuneScene(model); scene.add(model); document.getElementById('loading').style.display='none'; window.__FOUR_VEHICLE_READY=true; fitCamera(); }, progress => { if(progress.total) document.getElementById('loading').textContent = `loading four-vehicle reconstruction ${Math.round(progress.loaded/progress.total*100)}%`; }, err => { console.error(err); document.getElementById('loading').textContent='failed to load four-vehicle background GLB'; });

    const displayLabel = `${ultraQuality ? 'ultra ' : ''}four-agent static LiDAR background, ${(ultraQuality ? data.static_scene.ultra_point_count : data.static_scene.point_count).toLocaleString()} points; 4 moving vehicles`;
    document.getElementById('meta').innerHTML = `
      <span>Scenario</span><strong>${data.category}/${data.scenario}</strong>
      <span>Agents</span><strong>4 vehicles: colliding ego/other + 2 followers</strong>
      <span>Collision pair</span><strong>${data.collision?.mapped_vehicle_names?.join(' ↔ ') ?? data.source.meta_colliding_agents?.join(' / ')} · closest/overlap frame ${String((data.collision?.closest_or_overlap_frame?.source_frame ?? data.source.viewer_frame_count)).padStart(3,'0')} · OBB gap ${(data.collision?.closest_or_overlap_frame?.obb_gap_m ?? 0).toFixed(2)} m</strong>
      <span>Background</span><strong>static 3D reconstruction from all four DeepAccident LiDAR streams</strong>
      <span>Vehicle motion</span><strong>frame-wise DeepAccident calibration poses; selected scene has ego/other OBB overlap</strong>
      <span>Dynamic cleanup</span><strong>${Object.values(data.static_scene.removed_dynamic_vehicle_point_counts).reduce((a,b)=>a+b,0).toLocaleString()} vehicle points removed from background</strong>
      <span>Display</span><strong>${displayLabel}</strong>`;
    document.getElementById('legend').innerHTML = data.sequences.map(seq => `<span><i class="swatch" style="background:${seq.color}"></i>${seq.name}<em class="badge">${seq.vehicle_type}</em></span>`).join('');
    document.getElementById('frame').max = maxFrame;

    function setFrame(idx){ idx=Math.max(0,Math.min(maxFrame,Number(idx))); document.getElementById('frame').value=idx; document.getElementById('frameLabel').textContent=`frame ${String(idx+1).padStart(3,'0')}`;
      for(const seq of data.sequences){ const f=seq.frames[idx]; const v=vehicles.get(seq.key); if(v) applyPose(v,f.vehicle_matrix); const m=markers.get(seq.key); m.marker.position.copy(pathPosition(f)); updateActivePath(seq,idx); }
      document.getElementById('cams').innerHTML = data.sequences.map(seq => { const path=seq.frame_pattern.replace('{index4}',String(idx).padStart(4,'0')).replace('{index}',String(idx)); return `<figure class="cam"><img src="${path}" alt="${seq.name} frame ${idx+1}"><figcaption><strong>${seq.name}</strong><span>${seq.agent} ${String(idx+1).padStart(3,'0')}</span></figcaption></figure>`; }).join('');
      document.getElementById('stats').innerHTML = data.sequences.map(seq => { const f=seq.frames[idx]; return `<div class="stat"><b style="color:${seq.color}">${seq.name}</b><br>${seq.vehicle_type}, ${seq.role}<br>cumulative ${f.cumulative_motion.toFixed(2)} m · step ${f.step_motion.toFixed(2)} m</div>`; }).join('');
    }

    let playing=false,lastTick=0; document.getElementById('play').addEventListener('click',()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play'}); document.getElementById('reset').addEventListener('click',()=>{playing=false;document.getElementById('play').textContent='Play';fitCamera();setFrame(accidentIndex)}); document.getElementById('frame').addEventListener('input',e=>setFrame(e.target.value));
    setFrame(accidentIndex); fitCamera(); window.__FOUR_VEHICLE_CAPTURE=()=>canvas.toDataURL('image/png');
    window.addEventListener('resize',()=>{camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth,window.innerHeight)});
    function animate(t){ if(playing && t-lastTick>115){ const s=document.getElementById('frame'); setFrame(Number(s.value)>=maxFrame?0:Number(s.value)+1); lastTick=t; } controls.update(); renderer.render(scene,camera); requestAnimationFrame(animate); }
    animate(0);
  </script>
</body>
</html>
'''
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'index.html').write_text(html)


def build_data() -> dict:
    meta_info = parse_meta_info()
    counts = [frame_count(a['agent'], a['camera']) for a in AGENTS]
    frame_total = min(counts)
    origin = shared_origin(frame_total)
    sequences = build_vehicle_sequences(origin, frame_total)
    all_ground = []
    for agent_spec in AGENTS:
        for frame in range(1, frame_total + 1):
            calib = load_calib(agent_spec['agent'], frame)
            all_ground.append(float(np.asarray(calib['ego_to_world'])[2, 3] - origin[1]))
    ground_y = float(np.median(all_ground))
    bounds = trajectory_bounds(sequences, margin=24.0)
    # Keep enough vertical/side context for buildings and road furniture, but avoid enormous unrelated map areas.
    bounds['min'][1] = ground_y - 0.6
    bounds['max'][1] = ground_y + 13.0
    data = {
        'scenario': SCENARIO,
        'category': CATEGORY,
        'weather': meta_info.get('weather'),
        'road_type': meta_info.get('road_type'),
        'source': {
            'dataset': str(DATASET),
            'viewer_frame_count': frame_total,
            'viewer_accident_index': frame_total - 1,
            'metadata_accident_frame': meta_info.get('metadata_accident_frame'),
            'meta_agents_id_order': ['ego_vehicle', 'ego_vehicle_behind', 'other_vehicle', 'other_vehicle_behind'],
            'meta_colliding_agents': meta_info.get('colliding_agents'),
            'meta_agent_ids': meta_info.get('agent_ids'),
            'note': 'This rebuild selects Town03 because meta says colliding_agents=ego other, so the collision is between two of the four modeled vehicles.',
        },
        'metric_alignment': {
            'enabled': True,
            'origin_viewer_xyz': [float(v) for v in origin],
            'ground_y': ground_y,
            'coordinate_system': 'viewer X=CARLA X, viewer Y=CARLA Z/up, viewer Z=CARLA Y; all four vehicles share one metric world origin',
        },
        'sequences': sequences,
        'trajectory_bounds': bounds,
        'scene_bounds': bounds,
    }
    data['collision'] = build_collision_diagnostics(sequences, meta_info)
    data['static_scene'] = build_static_lidar_scene(data, frame_total)
    return data


def configure_paths(args: argparse.Namespace) -> None:
    global ROOT, DATASET, SOURCE_VIEWER, CATEGORY, SCENARIO, OUT, ASSETS, FRAMES, PREVIEWS

    ROOT = Path(args.output_root).expanduser().resolve()
    DATASET = Path(args.dataset).expanduser().resolve()
    SOURCE_VIEWER = Path(args.source_viewer).expanduser() if args.source_viewer else SOURCE_VIEWER
    CATEGORY = args.category
    SCENARIO = args.scenario
    OUT = ROOT / 'viewer'
    ASSETS = ROOT / 'viewer_assets'
    FRAMES = ROOT / 'viewer_frames'
    PREVIEWS = ROOT / 'previews'


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build the ReSceneScribe four-vehicle DeepAccident web replay.'
    )
    parser.add_argument('--dataset', default=str(DATASET), help='DeepAccident dataset root. Env: DEEPACCIDENT_ROOT')
    parser.add_argument('--output-root', default=str(ROOT), help='Directory where viewer/, viewer_assets/, viewer_frames/ are written.')
    parser.add_argument('--category', default=CATEGORY)
    parser.add_argument('--scenario', default=SCENARIO)
    parser.add_argument('--source-viewer', default=str(SOURCE_VIEWER) if SOURCE_VIEWER else '', help='Optional viewer directory whose vendor/three files can be reused.')
    args = parser.parse_args()
    configure_paths(args)

    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    counts = [frame_count(a['agent'], a['camera']) for a in AGENTS]
    frame_total = min(counts)
    copy_dashcam_frames(frame_total)
    data = build_data()
    make_topdown_preview(data)
    write_scene_data(data)
    copy_three_vendor()
    write_html()
    (OUT / 'README.md').write_text(
        '# DeepAccident Four Vehicle Collision Scene\n\n'
        'New viewer generated from the four DeepAccident Town03 moving agents. '\
        'The static background is a four-agent LiDAR fusion with points inside moving-vehicle boxes removed; '\
        'the four vehicles move using frame-wise calibration poses; ego and other collide in this scene.\n\n'
        'Open `index.html` through a local HTTP server.\n'
    )
    print(json.dumps({
        'viewer': str(OUT / 'index.html'),
        'frame_count': data['source']['viewer_frame_count'],
        'vehicles': [s['agent'] for s in data['sequences']],
        'static_scene': data['static_scene'],
    }, indent=2))


if __name__ == '__main__':
    main()
