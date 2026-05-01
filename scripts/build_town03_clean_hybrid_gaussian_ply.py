#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

DATASET = Path(os.environ.get('DEEPACCIDENT_ROOT', '/home/elicer/deepaccident_mini_dataset')).expanduser().resolve()
CATEGORY = os.environ.get('DEEPACCIDENT_CATEGORY', 'type1_subtype1_accident')
SCENARIO = os.environ.get('DEEPACCIDENT_SCENARIO', 'Town03_type001_subtype0001_scenario00024')
AGENTS = ['ego_vehicle', 'ego_vehicle_behind', 'other_vehicle', 'other_vehicle_behind']
ALL_CAMERAS = ['Camera_Front', 'Camera_FrontLeft', 'Camera_FrontRight', 'Camera_BackLeft', 'Camera_BackRight', 'Camera_Back']
DYNAMIC_TYPES = {'car', 'van', 'truck', 'bus', 'motorcycle', 'bicycle', 'cyclist', 'pedestrian', 'person'}
C0 = 0.28209479177387814


def base(agent: str, sensor: str) -> Path:
    return DATASET / CATEGORY / agent / sensor / SCENARIO


def load_calib(agent: str, frame: int) -> dict:
    with (base(agent, 'calib') / f'{SCENARIO}_{frame:03d}.pkl').open('rb') as f:
        return pickle.load(f)


def read_labels(agent: str, frame: int) -> list[dict]:
    path = base(agent, 'label') / f'{SCENARIO}_{frame:03d}.txt'
    labels: list[dict] = []
    if not path.exists():
        return labels
    for line in path.read_text(errors='replace').splitlines()[1:]:
        p = line.split()
        if len(p) < 13:
            continue
        try:
            labels.append({
                'type': p[0].lower(),
                'center': np.array([float(p[1]), float(p[2]), float(p[3])], dtype=np.float64),
                'dims': np.array([float(p[4]), float(p[5]), float(p[6])], dtype=np.float64),
                'yaw': float(p[7]),
                'id': int(float(p[10])),
                'raw': p,
            })
        except Exception:
            pass
    return labels


def read_self_dims(agent: str, frame: int = 1) -> dict:
    for label in read_labels(agent, frame):
        if label['id'] == -100:
            l, w, h = label['dims']
            return {'length': float(l), 'width': float(w), 'height': float(h)}
    raise RuntimeError(f'missing self dims for {agent}')


def vehicle_matrix(agent: str, frame: int) -> np.ndarray:
    return np.asarray(load_calib(agent, frame)['ego_to_world'], dtype=np.float64)


def world_to_vehicle_local(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    center = mat[:3, 3]
    rot = mat[:3, :3]
    return (points - center) @ rot


def vehicle_local_to_world(local: np.ndarray, mat: np.ndarray) -> np.ndarray:
    center = mat[:3, 3]
    rot = mat[:3, :3]
    return local @ rot.T + center


def inside_vehicle_world(points: np.ndarray, mat: np.ndarray, dims: dict, padding: float = 0.28, keep_road: bool = True) -> np.ndarray:
    local = world_to_vehicle_local(points, mat)
    z_min = 0.10 if keep_road else -0.45
    return (
        (np.abs(local[:, 0]) <= dims['length'] * 0.5 + padding)
        & (np.abs(local[:, 1]) <= dims['width'] * 0.5 + padding)
        & (local[:, 2] >= z_min)
        & (local[:, 2] <= dims['height'] + padding)
    )


def inside_label_lidar(points_lidar: np.ndarray, label: dict, padding: float = 0.30, remove_self: bool = True) -> np.ndarray:
    if label['id'] == -100 and not remove_self:
        return np.zeros(len(points_lidar), dtype=bool)
    l, w, h = label['dims']
    yaw = label['yaw']
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    local = (points_lidar[:, :3] - label['center']) @ R
    return (
        (np.abs(local[:, 0]) <= l * 0.5 + padding)
        & (np.abs(local[:, 1]) <= w * 0.5 + padding)
        & (local[:, 2] >= -h * 0.5 + 0.08)
        & (local[:, 2] <= h * 0.5 + padding)
    )


def dynamic_mask_lidar(points_lidar: np.ndarray, labels: list[dict]) -> np.ndarray:
    mask = np.zeros(len(points_lidar), dtype=bool)
    for label in labels:
        if label['id'] == -100 or label['type'] in DYNAMIC_TYPES:
            mask |= inside_label_lidar(points_lidar, label, remove_self=True)
    return mask


def trajectory_bounds(frame_start: int, frame_end: int, margin_xy: float = 46.0) -> dict:
    centers = []
    for f in range(frame_start, frame_end + 1):
        for a in AGENTS:
            centers.append(vehicle_matrix(a, f)[:3, 3])
    arr = np.asarray(centers)
    z_ground = float(np.median(arr[:, 2]))
    return {
        'min': np.array([arr[:, 0].min() - margin_xy, arr[:, 1].min() - margin_xy, z_ground - 2.0], dtype=np.float64),
        'max': np.array([arr[:, 0].max() + margin_xy, arr[:, 1].max() + margin_xy, z_ground + 18.0], dtype=np.float64),
        'ground_z': z_ground,
    }


def crop(points: np.ndarray, bounds: dict) -> np.ndarray:
    return np.all((points >= bounds['min']) & (points <= bounds['max']), axis=1)


def project_lidar_to_image(calib: dict, camera: str, pts_lidar: np.ndarray):
    hom = np.concatenate([pts_lidar[:, :3], np.ones((len(pts_lidar), 1), dtype=np.float64)], axis=1)
    l2c = np.asarray(calib[f'lidar_to_{camera}'], dtype=np.float64)
    K = np.asarray(calib[f'intrinsic_{camera}'], dtype=np.float64)
    cam = (l2c @ hom.T).T[:, :3]
    pr = (K @ cam.T).T
    den = pr[:, 2]
    u = pr[:, 0] / np.where(np.abs(den) > 1e-9, den, 1.0)
    v = pr[:, 1] / np.where(np.abs(den) > 1e-9, den, 1.0)
    return u, v, den


def lidar_colors(agent: str, frame: int, calib: dict, lidar: np.ndarray) -> np.ndarray:
    intensity = lidar[:, 3] if lidar.shape[1] > 3 else np.zeros(len(lidar), dtype=np.float64)
    lo, hi = np.percentile(intensity, 2), np.percentile(intensity, 98)
    norm = np.clip((intensity - lo) / max(hi - lo, 1e-6), 0, 1)
    colors = np.repeat((58 + norm[:, None] * 150).astype(np.uint8), 3, axis=1)
    assigned = np.zeros(len(lidar), dtype=bool)
    for cam in ALL_CAMERAS:
        img_path = base(agent, cam) / f'{SCENARIO}_{frame:03d}.jpg'
        if not img_path.exists():
            continue
        img = np.asarray(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]
        u, v, den = project_lidar_to_image(calib, cam, lidar[:, :3])
        finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(den) & (den > 0.35)
        # Only round pixels that are close enough to the image bounds to stay
        # inside after nearest-neighbour rounding.  This avoids overflow/cast
        # warnings from extreme projected values and prevents edge pixels from
        # rounding to exactly w/h.
        in_image = finite & (u >= -0.5) & (u < (w - 0.5)) & (v >= -0.5) & (v < (h - 0.5))
        ui = np.zeros(len(lidar), dtype=np.int32)
        vi = np.zeros(len(lidar), dtype=np.int32)
        ui[in_image] = np.rint(u[in_image]).astype(np.int32)
        vi[in_image] = np.rint(v[in_image]).astype(np.int32)
        valid = in_image & ~assigned
        if np.any(valid):
            colors[valid] = img[vi[valid], ui[valid]]
            assigned[valid] = True
        if assigned.all():
            break
    return colors


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel: float, limit: int, seed: int):
    if len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    points = points[idx]
    colors = colors[idx]
    if len(points) > limit:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(points), size=limit, replace=False))
        points = points[idx]
        colors = colors[idx]
    return points, colors


def build_cloud(frame_start: int, frame_end: int, static_limit: int, vehicle_limit_each: int, static_voxel: float, vehicle_voxel: float):
    bounds = trajectory_bounds(frame_start, frame_end)
    dims = {a: read_self_dims(a) for a in AGENTS}
    final_mats = {a: vehicle_matrix(a, frame_end) for a in AGENTS}

    static_pts: list[np.ndarray] = []
    static_cols: list[np.ndarray] = []
    vehicle_local: dict[str, list[np.ndarray]] = {a: [] for a in AGENTS}
    vehicle_cols: dict[str, list[np.ndarray]] = {a: [] for a in AGENTS}
    stats = {
        'source_lidar_points': {a: 0 for a in AGENTS},
        'static_points_before_downsample': 0,
        'removed_dynamic_points': {a: 0 for a in AGENTS},
        'vehicle_observation_points_before_downsample': {a: 0 for a in AGENTS},
        'vehicle_points_after_downsample': {},
    }

    for f in range(frame_start, frame_end + 1):
        vehicle_mats = {a: vehicle_matrix(a, f) for a in AGENTS}
        for sensor_agent in AGENTS:
            lidar_path = base(sensor_agent, 'lidar01') / f'{SCENARIO}_{f:03d}.npz'
            if not lidar_path.exists():
                continue
            calib = load_calib(sensor_agent, f)
            labels = read_labels(sensor_agent, f)
            lidar = np.load(lidar_path)['data'].astype(np.float64)
            stats['source_lidar_points'][sensor_agent] += int(len(lidar))
            colors = lidar_colors(sensor_agent, f, calib, lidar)
            hom = np.concatenate([lidar[:, :3], np.ones((len(lidar), 1), dtype=np.float64)], axis=1)
            world = (
                np.asarray(calib['ego_to_world'], dtype=np.float64)
                @ np.asarray(calib['lidar_to_ego'], dtype=np.float64)
                @ hom.T
            ).T[:, :3]

            dyn_local_mask = dynamic_mask_lidar(lidar[:, :3], labels)
            stats['removed_dynamic_points'][sensor_agent] += int(dyn_local_mask.sum())
            # Static background: remove all dynamic labeled objects and the four target vehicles.
            target_vehicle_mask = np.zeros(len(world), dtype=bool)
            for target in AGENTS:
                target_vehicle_mask |= inside_vehicle_world(world, vehicle_mats[target], dims[target], padding=0.38, keep_road=True)
            keep_static = (~dyn_local_mask) & (~target_vehicle_mask) & crop(world, bounds)
            if np.any(keep_static):
                static_pts.append(world[keep_static].astype(np.float32))
                static_cols.append(colors[keep_static].astype(np.uint8))

            # Dynamic vehicles: motion-compensate every observation into each vehicle's final accident pose.
            for target in AGENTS:
                vm = vehicle_mats[target]
                in_vehicle = inside_vehicle_world(world, vm, dims[target], padding=0.42, keep_road=True)
                if not np.any(in_vehicle):
                    continue
                local = world_to_vehicle_local(world[in_vehicle], vm)
                # Keep body surface only; suppress asphalt caught under the box.
                body = (
                    (np.abs(local[:, 0]) <= dims[target]['length'] * 0.5 + 0.35)
                    & (np.abs(local[:, 1]) <= dims[target]['width'] * 0.5 + 0.35)
                    & (local[:, 2] >= 0.05)
                    & (local[:, 2] <= dims[target]['height'] + 0.45)
                )
                if np.any(body):
                    vehicle_local[target].append(local[body].astype(np.float32))
                    vehicle_cols[target].append(colors[in_vehicle][body].astype(np.uint8))
                    stats['vehicle_observation_points_before_downsample'][target] += int(body.sum())

    if not static_pts:
        raise RuntimeError('no static points')
    spts = np.concatenate(static_pts, axis=0)
    scols = np.concatenate(static_cols, axis=0)
    stats['static_points_before_downsample'] = int(len(spts))
    spts, scols = voxel_downsample(spts, scols, static_voxel, static_limit, seed=1001)

    all_pts = [spts]
    all_cols = [scols]
    class_ids = [np.zeros(len(spts), dtype=np.uint8)]
    per_vehicle_final = {}
    for i, target in enumerate(AGENTS, start=1):
        if vehicle_local[target]:
            vloc = np.concatenate(vehicle_local[target], axis=0)
            vcols = np.concatenate(vehicle_cols[target], axis=0)
            vloc, vcols = voxel_downsample(vloc, vcols, vehicle_voxel, vehicle_limit_each, seed=2000 + i)
            vworld = vehicle_local_to_world(vloc.astype(np.float64), final_mats[target]).astype(np.float32)
            all_pts.append(vworld)
            all_cols.append(vcols)
            class_ids.append(np.full(len(vworld), i, dtype=np.uint8))
            per_vehicle_final[target] = int(len(vworld))
        else:
            per_vehicle_final[target] = 0
    stats['static_points_after_downsample'] = int(len(spts))
    stats['vehicle_points_after_downsample'] = per_vehicle_final
    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    cols = np.concatenate(all_cols, axis=0).astype(np.uint8)
    cls = np.concatenate(class_ids, axis=0)
    stats['total_points'] = int(len(pts))
    return pts, cols, cls, stats, bounds


def sigmoid_inv(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def write_hybrid_ply(path: Path, points: np.ndarray, colors: np.ndarray, cls: np.ndarray, static_scale: float, vehicle_scale: float) -> None:
    n = len(points)
    rgb = colors.astype(np.float32) / 255.0
    f_dc = (rgb - 0.5) / C0
    scales = np.zeros((n, 3), dtype=np.float32)
    scales[:] = math.log(static_scale)
    vehicle = cls > 0
    scales[vehicle] = math.log(vehicle_scale)
    opacity = np.full(n, sigmoid_inv(np.array([0.86], dtype=np.float32))[0], dtype=np.float32)
    opacity[vehicle] = sigmoid_inv(np.array([0.92], dtype=np.float32))[0]
    rots = np.zeros((n, 4), dtype=np.float32)
    rots[:, 0] = 1.0
    rest = np.zeros((n, 45), dtype=np.float32)
    normals = np.zeros((n, 3), dtype=np.float32)

    # Keep Graphdeco/3DGS fields in the standard order first.
    # Append RGB at the end so generic PLY viewers can still colorize by name.
    dtype = [('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),
             ('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4')]
    dtype += [(f'f_rest_{i}','f4') for i in range(45)]
    dtype += [('opacity','f4'),('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4'),
              ('red','u1'),('green','u1'),('blue','u1')]
    arr = np.empty(n, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = points[:,0], points[:,1], points[:,2]
    arr['nx'], arr['ny'], arr['nz'] = normals[:,0], normals[:,1], normals[:,2]
    arr['f_dc_0'], arr['f_dc_1'], arr['f_dc_2'] = f_dc[:,0], f_dc[:,1], f_dc[:,2]
    for i in range(45):
        arr[f'f_rest_{i}'] = rest[:, i]
    arr['opacity'] = opacity
    arr['scale_0'], arr['scale_1'], arr['scale_2'] = scales[:,0], scales[:,1], scales[:,2]
    arr['rot_0'], arr['rot_1'], arr['rot_2'], arr['rot_3'] = rots[:,0], rots[:,1], rots[:,2], rots[:,3]
    arr['red'], arr['green'], arr['blue'] = colors[:,0], colors[:,1], colors[:,2]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(path)


def configure_paths(args: argparse.Namespace) -> None:
    global DATASET, CATEGORY, SCENARIO

    DATASET = Path(args.dataset).expanduser().resolve()
    CATEGORY = args.category
    SCENARIO = args.scenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=str(DATASET), help='DeepAccident dataset root. Env: DEEPACCIDENT_ROOT')
    ap.add_argument('--category', default=CATEGORY)
    ap.add_argument('--scenario', default=SCENARIO)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frame-start', type=int, default=1)
    ap.add_argument('--frame-end', type=int, default=56)
    ap.add_argument('--static-limit', type=int, default=1100000)
    ap.add_argument('--vehicle-limit-each', type=int, default=120000)
    ap.add_argument('--static-voxel', type=float, default=0.050)
    ap.add_argument('--vehicle-voxel', type=float, default=0.025)
    ap.add_argument('--static-scale', type=float, default=0.040)
    ap.add_argument('--vehicle-scale', type=float, default=0.025)
    ap.add_argument('--stats', default='')
    args = ap.parse_args()
    configure_paths(args)
    points, colors, cls, stats, bounds = build_cloud(args.frame_start, args.frame_end, args.static_limit, args.vehicle_limit_each, args.static_voxel, args.vehicle_voxel)
    out = Path(args.out)
    write_hybrid_ply(out, points, colors, cls, args.static_scale, args.vehicle_scale)
    xyz = points.astype(np.float64)
    report = {
        'path': str(out),
        'scenario': SCENARIO,
        'frames': [args.frame_start, args.frame_end],
        'agents': AGENTS,
        'method': 'fused LiDAR geometry + dashcam RGB; static background dynamic masks; vehicle points motion-compensated into final accident poses; RGB + 3DGS hybrid PLY',
        'stats': stats,
        'bbox_min': xyz.min(axis=0).tolist(),
        'bbox_max': xyz.max(axis=0).tolist(),
        'bounds_used': {'min': bounds['min'].tolist(), 'max': bounds['max'].tolist(), 'ground_z': bounds['ground_z']},
        'ply_fields': 'standard RGB fields plus 3DGS f_dc/f_rest/opacity/scale/rot fields',
    }
    if args.stats:
        Path(args.stats).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
