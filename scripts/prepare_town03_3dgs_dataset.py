#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

DATASET = Path(os.environ.get('DEEPACCIDENT_ROOT', '/home/elicer/deepaccident_mini_dataset')).expanduser().resolve()
CATEGORY = os.environ.get('DEEPACCIDENT_CATEGORY', 'type1_subtype1_accident')
SCENARIO = os.environ.get('DEEPACCIDENT_SCENARIO', 'Town03_type001_subtype0001_scenario00024')
AGENTS = ['ego_vehicle', 'ego_vehicle_behind', 'other_vehicle', 'other_vehicle_behind']
CAMERA = 'Camera_Front'
ALL_CAMERAS = ['Camera_Front', 'Camera_FrontLeft', 'Camera_FrontRight', 'Camera_BackLeft', 'Camera_BackRight', 'Camera_Back']
WIDTH = 1600
HEIGHT = 900
FX = FY = 1142.5184053936917
CX = 800.0
CY = 450.0
# CARLA camera coordinates are X-forward, Y-right, Z-up. COLMAP/3DGS expects X-right, Y-down, Z-forward.
CARLA_CAM_TO_COLMAP_CAM = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)


def base(agent: str, sensor: str) -> Path:
    return DATASET / CATEGORY / agent / sensor / SCENARIO


def load_calib(agent: str, frame: int) -> dict:
    with (base(agent, 'calib') / f'{SCENARIO}_{frame:03d}.pkl').open('rb') as f:
        return pickle.load(f)


def read_self_dims(agent: str, frame: int = 1) -> dict:
    label = base(agent, 'label') / f'{SCENARIO}_{frame:03d}.txt'
    for line in label.read_text(errors='replace').splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 11 and parts[10] == '-100':
            return {
                'type': parts[0],
                'length': float(parts[4]),
                'width': float(parts[5]),
                'height': float(parts[6]),
            }
    raise RuntimeError(f'No self label in {label}')


def vehicle_matrix_carla(agent: str, frame: int) -> np.ndarray:
    return np.asarray(load_calib(agent, frame)['ego_to_world'], dtype=np.float64)


def camera_c2w_colmap(agent: str, frame: int) -> np.ndarray:
    c = load_calib(agent, frame)
    c2w_carla_cam = (
        np.asarray(c['ego_to_world'], dtype=np.float64)
        @ np.asarray(c['lidar_to_ego'], dtype=np.float64)
        @ np.linalg.inv(np.asarray(c[f'lidar_to_{CAMERA}'], dtype=np.float64))
    )
    s_inv = np.eye(4, dtype=np.float64)
    s_inv[:3, :3] = CARLA_CAM_TO_COLMAP_CAM.T
    return c2w_carla_cam @ s_inv


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_colmap_text(dataset_dir: Path, frame_start: int, frame_end: int) -> list[dict]:
    sparse = dataset_dir / 'sparse' / '0'
    images = dataset_dir / 'images'
    sparse.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    (sparse / 'cameras.txt').write_text(
        '# Camera list with one line of data per camera:\n'
        '# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n'
        '# Number of cameras: 1\n'
        f'1 PINHOLE {WIDTH} {HEIGHT} {FX:.12f} {FY:.12f} {CX:.12f} {CY:.12f}\n'
    )

    image_rows = []
    image_id = 1
    for frame in range(frame_start, frame_end + 1):
        for agent in AGENTS:
            src = base(agent, CAMERA) / f'{SCENARIO}_{frame:03d}.jpg'
            name = f'{agent}_front_{frame:03d}.jpg'
            dst = images / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)

            c2w = camera_c2w_colmap(agent, frame)
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            q = rotmat2qvec(R)
            image_rows.append({
                'image_id': image_id,
                'agent': agent,
                'frame': frame,
                'name': name,
                'qvec': q.tolist(),
                'tvec': t.tolist(),
                'camera_center': c2w[:3, 3].tolist(),
            })
            image_id += 1

    lines = [
        '# Image list with two lines of data per image:\n',
        '# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n',
        '# POINTS2D[] as (X, Y, POINT3D_ID)\n',
        f'# Number of images: {len(image_rows)}, mean observations per image: 0\n',
    ]
    for row in image_rows:
        q = row['qvec']
        t = row['tvec']
        lines.append(
            f"{row['image_id']} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
            f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} 1 {row['name']}\n"
        )
        lines.append('\n')
    (sparse / 'images.txt').write_text(''.join(lines))
    return image_rows


def trajectory_bounds(frame_min: int, frame_max: int, margin_xy: float = 46.0) -> dict:
    centers = []
    for frame in range(frame_min, frame_max + 1):
        for agent in AGENTS:
            centers.append(vehicle_matrix_carla(agent, frame)[:3, 3])
    arr = np.asarray(centers)
    z_ground = float(np.median(arr[:, 2]))
    return {
        'min': [float(arr[:, 0].min() - margin_xy), float(arr[:, 1].min() - margin_xy), z_ground - 1.2],
        'max': [float(arr[:, 0].max() + margin_xy), float(arr[:, 1].max() + margin_xy), z_ground + 22.0],
        'ground_z': z_ground,
    }


def points_inside_vehicle(points: np.ndarray, matrix: np.ndarray, dims: dict, padding: float = 0.42, keep_road: bool = True) -> np.ndarray:
    center = matrix[:3, 3]
    rot = matrix[:3, :3]
    local = (points - center) @ rot
    z_min = 0.10 if keep_road else -0.55
    return (
        (np.abs(local[:, 0]) <= dims['length'] * 0.5 + padding)
        & (np.abs(local[:, 1]) <= dims['width'] * 0.5 + padding)
        & (local[:, 2] >= z_min)
        & (local[:, 2] <= dims['height'] + padding)
    )


def project_lidar_colors(agent: str, frame: int, calib: dict, lidar_points: np.ndarray) -> np.ndarray:
    intensity = lidar_points[:, 3] if lidar_points.shape[1] > 3 else np.zeros(len(lidar_points), dtype=np.float64)
    lo, hi = np.percentile(intensity, 2), np.percentile(intensity, 98)
    norm = np.clip((intensity - lo) / max(hi - lo, 1e-6), 0, 1)
    colors = np.repeat((65 + norm[:, None] * 145).astype(np.uint8), 3, axis=1)
    assigned = np.zeros(len(lidar_points), dtype=bool)
    homogeneous = np.concatenate([lidar_points[:, :3], np.ones((len(lidar_points), 1), dtype=np.float64)], axis=1)
    for camera in ALL_CAMERAS:
        img_path = base(agent, camera) / f'{SCENARIO}_{frame:03d}.jpg'
        if not img_path.exists():
            continue
        img = np.asarray(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]
        lidar_to_camera = np.asarray(calib[f'lidar_to_{camera}'], dtype=np.float64)
        intrinsic = np.asarray(calib[f'intrinsic_{camera}'], dtype=np.float64)
        cam_points = (lidar_to_camera @ homogeneous.T).T[:, :3]
        projected = (intrinsic @ cam_points.T).T
        denom = projected[:, 2]
        valid_depth = denom > 1e-3
        u = projected[:, 0] / np.where(valid_depth, denom, 1.0)
        v = projected[:, 1] / np.where(valid_depth, denom, 1.0)
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        valid = valid_depth & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & ~assigned
        if np.any(valid):
            colors[valid] = img[vi[valid], ui[valid]]
            assigned[valid] = True
        if assigned.all():
            break
    return colors


def crop_mask(points: np.ndarray, bounds: dict) -> np.ndarray:
    mn, mx = bounds['min'], bounds['max']
    return (
        (points[:, 0] >= mn[0]) & (points[:, 0] <= mx[0])
        & (points[:, 1] >= mn[1]) & (points[:, 1] <= mx[1])
        & (points[:, 2] >= mn[2]) & (points[:, 2] <= mx[2])
    )


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, keep_idx = np.unique(keys, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    points = points[keep_idx]
    colors = colors[keep_idx]
    if len(points) > limit:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(points), size=limit, replace=False))
        points = points[idx]
        colors = colors[idx]
    return points, colors


def build_initial_ply(dataset_dir: Path, frame_start: int, frame_end: int, final_frame: int, static_limit: int, vehicle_limit: int) -> dict:
    bounds = trajectory_bounds(1, final_frame)
    dims = {agent: read_self_dims(agent) for agent in AGENTS}
    static_points = []
    static_colors = []
    final_vehicle_points = []
    final_vehicle_colors = []
    source_counts = {agent: 0 for agent in AGENTS}
    kept_static_counts = {agent: 0 for agent in AGENTS}
    kept_final_vehicle_counts = {agent: 0 for agent in AGENTS}

    # Static background uses all frames up to the accident, with all moving vehicle boxes removed.
    for frame in range(1, final_frame + 1):
        boxes = [(vehicle_matrix_carla(agent, frame), dims[agent]) for agent in AGENTS]
        final_boxes = [(vehicle_matrix_carla(agent, final_frame), dims[agent]) for agent in AGENTS]
        for agent in AGENTS:
            lidar_path = base(agent, 'lidar01') / f'{SCENARIO}_{frame:03d}.npz'
            if not lidar_path.exists():
                continue
            calib = load_calib(agent, frame)
            lidar = np.load(lidar_path)['data'].astype(np.float64)
            source_counts[agent] += int(len(lidar))
            hom = np.concatenate([lidar[:, :3], np.ones((len(lidar), 1), dtype=np.float64)], axis=1)
            world = (np.asarray(calib['ego_to_world'], dtype=np.float64) @ np.asarray(calib['lidar_to_ego'], dtype=np.float64) @ hom.T).T[:, :3]
            colors = project_lidar_colors(agent, frame, calib, lidar)
            keep = crop_mask(world, bounds)
            if np.any(keep):
                pts = world[keep]
                cols = colors[keep]
                dynamic = np.zeros(len(pts), dtype=bool)
                for m, d in boxes:
                    dynamic |= points_inside_vehicle(pts, m, d, keep_road=True)
                static_keep = ~dynamic
                if np.any(static_keep):
                    static_points.append(pts[static_keep].astype(np.float32))
                    static_colors.append(cols[static_keep].astype(np.uint8))
                    kept_static_counts[agent] += int(static_keep.sum())

            # Add final accident-state vehicle surfaces only from the final frame.
            if frame == final_frame and np.any(keep):
                pts = world[keep]
                cols = colors[keep]
                final_dyn = np.zeros(len(pts), dtype=bool)
                for m, d in final_boxes:
                    final_dyn |= points_inside_vehicle(pts, m, d, padding=0.55, keep_road=True)
                if np.any(final_dyn):
                    final_vehicle_points.append(pts[final_dyn].astype(np.float32))
                    final_vehicle_colors.append(cols[final_dyn].astype(np.uint8))
                    kept_final_vehicle_counts[agent] += int(final_dyn.sum())

    if not static_points:
        raise RuntimeError('No static initialization points')
    static_p = np.concatenate(static_points, axis=0)
    static_c = np.concatenate(static_colors, axis=0)
    static_p, static_c = voxel_downsample(static_p, static_c, voxel_size=0.070, limit=static_limit, seed=66)

    if final_vehicle_points:
        veh_p = np.concatenate(final_vehicle_points, axis=0)
        veh_c = np.concatenate(final_vehicle_colors, axis=0)
        veh_p, veh_c = voxel_downsample(veh_p, veh_c, voxel_size=0.035, limit=vehicle_limit, seed=67)
        points = np.concatenate([static_p, veh_p], axis=0)
        colors = np.concatenate([static_c, veh_c], axis=0)
    else:
        veh_p = np.zeros((0, 3), dtype=np.float32)
        points, colors = static_p, static_c

    sparse = dataset_dir / 'sparse' / '0'
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(points, dtype=np.float32)
    elements = np.empty(len(points), dtype=dtype)
    elements[:] = list(map(tuple, np.concatenate([points.astype(np.float32), normals, colors.astype(np.uint8)], axis=1)))
    PlyData([PlyElement.describe(elements, 'vertex')], text=False).write(sparse / 'points3D.ply')
    (sparse / 'points3D.txt').write_text('# points are stored in points3D.ply for 3DGS initialization\n')

    return {
        'bounds': bounds,
        'source_lidar_points': source_counts,
        'kept_static_before_downsample': kept_static_counts,
        'kept_final_vehicle_before_downsample': kept_final_vehicle_counts,
        'static_points_after_downsample': int(len(static_p)),
        'final_vehicle_points_after_downsample': int(len(veh_p)),
        'initial_points_total': int(len(points)),
        'ply': str(sparse / 'points3D.ply'),
    }


def main() -> None:
    global DATASET, CATEGORY, SCENARIO

    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=str(DATASET), help='DeepAccident dataset root. Env: DEEPACCIDENT_ROOT')
    ap.add_argument('--category', default=CATEGORY)
    ap.add_argument('--scenario', default=SCENARIO)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frame-start', type=int, default=41)
    ap.add_argument('--frame-end', type=int, default=56)
    ap.add_argument('--static-limit', type=int, default=650000)
    ap.add_argument('--vehicle-limit', type=int, default=140000)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    DATASET = Path(args.dataset).expanduser().resolve()
    CATEGORY = args.category
    SCENARIO = args.scenario
    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = write_colmap_text(out, args.frame_start, args.frame_end)
    init = build_initial_ply(out, args.frame_start, args.frame_end, args.frame_end, args.static_limit, args.vehicle_limit)
    meta = {
        'category': CATEGORY,
        'scenario': SCENARIO,
        'camera': CAMERA,
        'frame_start': args.frame_start,
        'frame_end': args.frame_end,
        'image_count': len(rows),
        'image_resolution': [WIDTH, HEIGHT],
        'intrinsics_pinhole': {'fx': FX, 'fy': FY, 'cx': CX, 'cy': CY},
        'coordinate_system': 'CARLA world coordinates; camera CARLA X-forward/Y-right/Z-up converted to COLMAP X-right/Y-down/Z-forward',
        'initialization': init,
        'collision_note': 'Town03 meta has colliding_agents ego other; selected frames end at the available collision/overlap frame 056.',
    }
    (out / 'dataset_manifest.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
