#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
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
# DeepAccident raw camera calibration projects CARLA-style camera coords:
#   X forward, Y right, Z up -> u = cx + f*Y/X, v = cy - f*Z/X.
# COLMAP/3DGS uses X right, Y down, Z forward.
CARLA_CAM_TO_COLMAP_CAM = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
DYNAMIC_TYPES = {'car', 'van', 'truck', 'bus', 'motorcycle', 'bicycle', 'cyclist', 'pedestrian', 'person'}


def base(agent: str, sensor: str) -> Path:
    return DATASET / CATEGORY / agent / sensor / SCENARIO


def load_calib(agent: str, frame: int) -> dict:
    with (base(agent, 'calib') / f'{SCENARIO}_{frame:03d}.pkl').open('rb') as f:
        return pickle.load(f)


def read_labels(agent: str, frame: int) -> list[dict]:
    path = base(agent, 'label') / f'{SCENARIO}_{frame:03d}.txt'
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(errors='replace').splitlines()[1:]:
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            obj_id = int(float(parts[10]))
            labels.append({
                'type': parts[0].lower(),
                'center': np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64),
                'dims': np.array([float(parts[4]), float(parts[5]), float(parts[6])], dtype=np.float64),
                'yaw': float(parts[7]),
                'id': obj_id,
                'raw': parts,
            })
        except Exception:
            continue
    return labels


def vehicle_matrix_carla(agent: str, frame: int) -> np.ndarray:
    return np.asarray(load_calib(agent, frame)['ego_to_world'], dtype=np.float64)


def camera_c2w_colmap(agent: str, frame: int) -> np.ndarray:
    c = load_calib(agent, frame)
    lidar_to_cam_carla = np.asarray(c[f'lidar_to_{CAMERA}'], dtype=np.float64)
    c2w_carla_cam = (
        np.asarray(c['ego_to_world'], dtype=np.float64)
        @ np.asarray(c['lidar_to_ego'], dtype=np.float64)
        @ np.linalg.inv(lidar_to_cam_carla)
    )
    colmap_to_carla = np.eye(4, dtype=np.float64)
    colmap_to_carla[:3, :3] = CARLA_CAM_TO_COLMAP_CAM.T
    return c2w_carla_cam @ colmap_to_carla


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


def box_corners_lidar(label: dict, scale: float = 1.08) -> np.ndarray:
    l, w, h = label['dims'] * scale
    xs = [-l / 2, l / 2]
    ys = [-w / 2, w / 2]
    zs = [-h / 2, h / 2]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    yaw = label['yaw']
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return corners @ R.T + label['center']


def project_lidar_to_image(calib: dict, camera: str, pts_lidar: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hom = np.concatenate([pts_lidar, np.ones((len(pts_lidar), 1), dtype=np.float64)], axis=1)
    lidar_to_camera = np.asarray(calib[f'lidar_to_{camera}'], dtype=np.float64)
    intrinsic = np.asarray(calib[f'intrinsic_{camera}'], dtype=np.float64)
    cam_pts = (lidar_to_camera @ hom.T).T[:, :3]
    proj = (intrinsic @ cam_pts.T).T
    denom = proj[:, 2]
    u = proj[:, 0] / np.where(np.abs(denom) > 1e-9, denom, 1.0)
    v = proj[:, 1] / np.where(np.abs(denom) > 1e-9, denom, 1.0)
    return u, v, denom


def mask_dynamic_pixels(img_path: Path, out_path: Path, calib: dict, labels: list[dict], pad_px: int = 12) -> int:
    img = Image.open(img_path).convert('RGBA')
    rgba = np.array(img)
    alpha = rgba[:, :, 3]
    draw = ImageDraw.Draw(img)
    masks = 0
    for label in labels:
        if label['id'] == -100 or label['type'] not in DYNAMIC_TYPES:
            continue
        corners = box_corners_lidar(label)
        u, v, den = project_lidar_to_image(calib, CAMERA, corners)
        valid = den > 0.35
        if valid.sum() < 2:
            continue
        uu = u[valid]
        vv = v[valid]
        if uu.max() < -200 or uu.min() > WIDTH + 200 or vv.max() < -200 or vv.min() > HEIGHT + 200:
            continue
        x0 = max(0, int(math.floor(uu.min())) - pad_px)
        y0 = max(0, int(math.floor(vv.min())) - pad_px)
        x1 = min(WIDTH - 1, int(math.ceil(uu.max())) + pad_px)
        y1 = min(HEIGHT - 1, int(math.ceil(vv.max())) + pad_px)
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            continue
        area = (x1 - x0) * (y1 - y0)
        # Almost-full-frame masks usually come from an object crossing the camera plane.
        # Keep large close-vehicle masks, but reject pathological projections.
        if area > WIDTH * HEIGHT * 0.88:
            continue
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 0))
        masks += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, compress_level=3)
    return masks


def write_colmap_text_and_images(dataset_dir: Path, frame_start: int, frame_end: int) -> tuple[list[dict], dict]:
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

    rows = []
    image_id = 1
    mask_counts = {agent: 0 for agent in AGENTS}
    for frame in range(frame_start, frame_end + 1):
        for agent in AGENTS:
            calib = load_calib(agent, frame)
            labels = read_labels(agent, frame)
            src = base(agent, CAMERA) / f'{SCENARIO}_{frame:03d}.jpg'
            name = f'{agent}_front_masked_{frame:03d}.png'
            dst = images / name
            mask_counts[agent] += mask_dynamic_pixels(src, dst, calib, labels)

            c2w = camera_c2w_colmap(agent, frame)
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            q = rotmat2qvec(R)
            rows.append({
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
        f'# Number of images: {len(rows)}, mean observations per image: 0\n',
    ]
    for row in rows:
        q = row['qvec']
        t = row['tvec']
        lines.append(
            f"{row['image_id']} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
            f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} 1 {row['name']}\n"
        )
        lines.append('\n')
    (sparse / 'images.txt').write_text(''.join(lines))
    return rows, mask_counts


def trajectory_bounds(frame_min: int, frame_max: int, margin_xy: float = 42.0) -> dict:
    centers = []
    for frame in range(frame_min, frame_max + 1):
        for agent in AGENTS:
            centers.append(vehicle_matrix_carla(agent, frame)[:3, 3])
    arr = np.asarray(centers)
    z_ground = float(np.median(arr[:, 2]))
    return {
        'min': [float(arr[:, 0].min() - margin_xy), float(arr[:, 1].min() - margin_xy), z_ground - 2.0],
        'max': [float(arr[:, 0].max() + margin_xy), float(arr[:, 1].max() + margin_xy), z_ground + 18.0],
        'ground_z': z_ground,
    }


def crop_mask(points: np.ndarray, bounds: dict) -> np.ndarray:
    mn, mx = np.asarray(bounds['min']), np.asarray(bounds['max'])
    return np.all((points >= mn) & (points <= mx), axis=1)


def points_inside_lidar_box(points_lidar: np.ndarray, label: dict, padding: float = 0.32) -> np.ndarray:
    l, w, h = label['dims']
    yaw = label['yaw']
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    local = (points_lidar[:, :3] - label['center']) @ R
    # Preserve the road under cars: remove car body, not the asphalt plane.
    return (
        (np.abs(local[:, 0]) <= l * 0.5 + padding)
        & (np.abs(local[:, 1]) <= w * 0.5 + padding)
        & (local[:, 2] >= -h * 0.5 + 0.12)
        & (local[:, 2] <= h * 0.5 + padding)
    )


def dynamic_lidar_mask(points_lidar: np.ndarray, labels: list[dict]) -> np.ndarray:
    mask = np.zeros(len(points_lidar), dtype=bool)
    for label in labels:
        if label['id'] == -100 or label['type'] not in DYNAMIC_TYPES:
            continue
        mask |= points_inside_lidar_box(points_lidar, label)
    return mask


def project_lidar_colors(agent: str, frame: int, calib: dict, lidar_points: np.ndarray) -> np.ndarray:
    intensity = lidar_points[:, 3] if lidar_points.shape[1] > 3 else np.zeros(len(lidar_points), dtype=np.float64)
    lo, hi = np.percentile(intensity, 2), np.percentile(intensity, 98)
    norm = np.clip((intensity - lo) / max(hi - lo, 1e-6), 0, 1)
    colors = np.repeat((55 + norm[:, None] * 155).astype(np.uint8), 3, axis=1)
    assigned = np.zeros(len(lidar_points), dtype=bool)
    for camera in ALL_CAMERAS:
        img_path = base(agent, camera) / f'{SCENARIO}_{frame:03d}.jpg'
        if not img_path.exists():
            continue
        img = np.asarray(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]
        u, v, den = project_lidar_to_image(calib, camera, lidar_points[:, :3])
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        valid = (den > 0.35) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & ~assigned
        if np.any(valid):
            colors[valid] = img[vi[valid], ui[valid]]
            assigned[valid] = True
        if assigned.all():
            break
    return colors


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


def build_static_initial_ply(dataset_dir: Path, frame_start: int, frame_end: int, static_limit: int, voxel: float) -> dict:
    bounds = trajectory_bounds(frame_start, frame_end)
    static_points = []
    static_colors = []
    source_counts = {agent: 0 for agent in AGENTS}
    kept_counts = {agent: 0 for agent in AGENTS}
    removed_dynamic_counts = {agent: 0 for agent in AGENTS}

    for frame in range(frame_start, frame_end + 1):
        for agent in AGENTS:
            lidar_path = base(agent, 'lidar01') / f'{SCENARIO}_{frame:03d}.npz'
            if not lidar_path.exists():
                continue
            calib = load_calib(agent, frame)
            labels = read_labels(agent, frame)
            lidar = np.load(lidar_path)['data'].astype(np.float64)
            source_counts[agent] += int(len(lidar))
            dyn = dynamic_lidar_mask(lidar[:, :3], labels)
            removed_dynamic_counts[agent] += int(dyn.sum())
            keep_local = ~dyn
            if not np.any(keep_local):
                continue
            lidar_kept = lidar[keep_local]
            hom = np.concatenate([lidar_kept[:, :3], np.ones((len(lidar_kept), 1), dtype=np.float64)], axis=1)
            world = (
                np.asarray(calib['ego_to_world'], dtype=np.float64)
                @ np.asarray(calib['lidar_to_ego'], dtype=np.float64)
                @ hom.T
            ).T[:, :3]
            keep_world = crop_mask(world, bounds)
            if not np.any(keep_world):
                continue
            cols = project_lidar_colors(agent, frame, calib, lidar_kept)
            static_points.append(world[keep_world].astype(np.float32))
            static_colors.append(cols[keep_world].astype(np.uint8))
            kept_counts[agent] += int(keep_world.sum())

    if not static_points:
        raise RuntimeError('No static initialization points were collected')
    points = np.concatenate(static_points, axis=0)
    colors = np.concatenate(static_colors, axis=0)
    before_downsample = int(len(points))
    points, colors = voxel_downsample(points, colors, voxel_size=voxel, limit=static_limit, seed=703)

    sparse = dataset_dir / 'sparse' / '0'
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(points, dtype=np.float32)
    elements = np.empty(len(points), dtype=dtype)
    arr = np.concatenate([points.astype(np.float32), normals, colors.astype(np.uint8)], axis=1)
    elements[:] = list(map(tuple, arr))
    PlyData([PlyElement.describe(elements, 'vertex')], text=False).write(sparse / 'points3D.ply')
    (sparse / 'points3D.txt').write_text('# points stored in points3D.ply for masked static 3DGS initialization\n')
    return {
        'bounds': bounds,
        'source_lidar_points': source_counts,
        'removed_dynamic_lidar_points': removed_dynamic_counts,
        'kept_static_before_downsample_by_agent': kept_counts,
        'static_points_before_downsample': before_downsample,
        'static_points_after_downsample': int(len(points)),
        'ply': str(sparse / 'points3D.ply'),
    }


def main() -> None:
    global DATASET, CATEGORY, SCENARIO

    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=str(DATASET), help='DeepAccident dataset root. Env: DEEPACCIDENT_ROOT')
    ap.add_argument('--category', default=CATEGORY)
    ap.add_argument('--scenario', default=SCENARIO)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frame-start', type=int, default=1)
    ap.add_argument('--frame-end', type=int, default=56)
    ap.add_argument('--static-limit', type=int, default=950000)
    ap.add_argument('--voxel', type=float, default=0.055)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    DATASET = Path(args.dataset).expanduser().resolve()
    CATEGORY = args.category
    SCENARIO = args.scenario
    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, mask_counts = write_colmap_text_and_images(out, args.frame_start, args.frame_end)
    init = build_static_initial_ply(out, args.frame_start, args.frame_end, args.static_limit, args.voxel)
    meta = {
        'category': CATEGORY,
        'scenario': SCENARIO,
        'camera': CAMERA,
        'agents': AGENTS,
        'frame_start': args.frame_start,
        'frame_end': args.frame_end,
        'image_count': len(rows),
        'image_resolution': [WIDTH, HEIGHT],
        'intrinsics_pinhole_colmap': {'fx': FX, 'fy': FY, 'cx': CX, 'cy': CY},
        'masking': {
            'dynamic_types': sorted(DYNAMIC_TYPES),
            'method': 'vehicle/person 3D label boxes projected to front dashcam; masked pixels are RGB=0, alpha=0 so vanilla 3DGS loss ignores them',
            'mask_boxes_by_agent': mask_counts,
        },
        'initialization': init,
        'coordinate_system': 'CARLA world; DeepAccident X-forward/Y-right/Z-up camera converted to COLMAP X-right/Y-down/Z-forward',
        'note': 'Static background reconstruction. Moving accident vehicles are intentionally masked out to avoid tearing/ghosting in static 3DGS.',
    }
    (out / 'dataset_manifest.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
