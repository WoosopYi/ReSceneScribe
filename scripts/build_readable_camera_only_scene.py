#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from plyfile import PlyData, PlyElement


AGENT_COLORS = {
    "ego_vehicle": "#ff4d4d",
    "ego_vehicle_behind": "#2dd4bf",
    "other_vehicle": "#facc15",
    "other_vehicle_behind": "#60a5fa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visually readable road-frame scene from camera-only outputs.")
    parser.add_argument("--source", type=Path, default=Path("outputs/multicam_world_reconstruction"))
    parser.add_argument("--out", type=Path, default=Path("outputs/multicam_world_reconstruction_readable"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--grid-resolution", type=float, default=0.25)
    parser.add_argument("--sample-step", type=int, default=16)
    parser.add_argument("--trajectory-margin", type=float, default=35.0)
    parser.add_argument("--max-vertical-points", type=int, default=220000)
    parser.add_argument("--height-min", type=float, default=0.45)
    parser.add_argument("--height-max", type=float, default=18.0)
    parser.add_argument("--include-vggt-points", action="store_true")
    parser.add_argument("--use-bev-background", action="store_true")
    parser.add_argument("--bev-agent", default="ego_vehicle")
    parser.add_argument("--bev-frame", type=int, default=46)
    parser.add_argument("--bev-rotation", type=int, default=0)
    parser.add_argument("--bev-flip-x", action="store_true")
    parser.add_argument("--bev-flip-y", action="store_true")
    parser.add_argument("--background-stride", type=int, default=1)
    parser.add_argument("--building-min-area-m2", type=float, default=18.0)
    parser.add_argument("--max-buildings", type=int, default=44)
    parser.add_argument("--max-trees", type=int, default=90)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def fit_road_frame(agent_tracks: dict, diagnostics: dict) -> dict:
    all_positions = []
    for agent in agent_tracks["agents"].values():
        for sample in agent["samples"]:
            all_positions.append(sample["position_world"])
    points = np.asarray(all_positions, dtype=np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)

    ego_samples = agent_tracks["agents"]["ego_vehicle"]["samples"]
    ego_start = np.asarray(ego_samples[0]["position_world"], dtype=np.float64)
    ego_end = np.asarray(ego_samples[-1]["position_world"], dtype=np.float64)
    forward = ego_end - ego_start
    forward -= normal * np.dot(forward, normal)
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, normal)
    right /= max(np.linalg.norm(right), 1e-9)
    forward = np.cross(normal, right)
    forward /= max(np.linalg.norm(forward), 1e-9)

    closest = diagnostics.get("closest_approach")
    if closest:
        frame = int(closest["frame"])
        pair = closest["agents"]
        positions = []
        for agent_name in pair:
            for sample in agent_tracks["agents"][agent_name]["samples"]:
                if int(sample["frame"]) == frame:
                    positions.append(sample["position_world"])
        origin = np.asarray(positions, dtype=np.float64).mean(axis=0) if positions else centroid
    else:
        origin = centroid
    origin = origin - normal * np.dot(origin - centroid, normal)

    return {
        "origin_world": origin,
        "right_world": right,
        "up_world": normal,
        "forward_world": forward,
        "basis_world_from_road": np.column_stack([right, normal, forward]),
    }


def world_to_road(points_world: np.ndarray, frame: dict) -> np.ndarray:
    basis = frame["basis_world_from_road"]
    return (np.asarray(points_world, dtype=np.float64) - frame["origin_world"]) @ basis


def road_to_world(points_road: np.ndarray, frame: dict) -> np.ndarray:
    basis = frame["basis_world_from_road"]
    return np.asarray(points_road, dtype=np.float64) @ basis.T + frame["origin_world"]


def trajectory_extents(agent_tracks: dict, frame: dict, margin: float) -> dict:
    road_positions = []
    for agent in agent_tracks["agents"].values():
        for sample in agent["samples"]:
            road_positions.append(world_to_road(np.asarray(sample["position_world"])[None, :], frame)[0])
    arr = np.asarray(road_positions, dtype=np.float64)
    return {
        "x_min": float(np.floor(arr[:, 0].min() - margin)),
        "x_max": float(np.ceil(arr[:, 0].max() + margin)),
        "z_min": float(np.floor(arr[:, 2].min() - margin)),
        "z_max": float(np.ceil(arr[:, 2].max() + margin)),
    }


def standard_intrinsics(raw_k: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    fx = abs(float(raw_k[0, 1])) if abs(float(raw_k[0, 1])) > 1.0 else abs(float(raw_k[1, 2]))
    fy = abs(float(raw_k[1, 2])) if abs(float(raw_k[1, 2])) > 1.0 else fx
    cx = float(raw_k[0, 0]) if abs(float(raw_k[0, 0])) > 1.0 else width / 2.0
    cy = float(raw_k[1, 0]) if abs(float(raw_k[1, 0])) > 1.0 else height / 2.0
    return fx, fy, cx, cy


def rasterize_ground_texture(source: Path, cameras: dict, frame: dict, extents: dict, resolution: float, sample_step: int) -> dict:
    width_cells = int(math.ceil((extents["x_max"] - extents["x_min"]) / resolution)) + 1
    height_cells = int(math.ceil((extents["z_max"] - extents["z_min"]) / resolution)) + 1
    rgb_sum = np.zeros((height_cells, width_cells, 3), dtype=np.float64)
    weight_sum = np.zeros((height_cells, width_cells), dtype=np.float64)
    hit_count = np.zeros((height_cells, width_cells), dtype=np.uint16)

    plane_origin = frame["origin_world"]
    plane_normal = frame["up_world"]
    views_used = 0
    rays_used = 0
    for view in cameras["views"]:
        image_path = source / view["rgb_path"]
        mask_path = source / view["mask_path"]
        if not image_path.exists():
            continue
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if mask_path.exists():
            mask = np.asarray(Image.open(mask_path).convert("L"))
        h, w = image.shape[:2]
        fx, fy, cx, cy = standard_intrinsics(np.asarray(view["intrinsic_raw"], dtype=np.float64), w, h)
        c2w = np.asarray(view["camera_to_world_cv"], dtype=np.float64)
        cam_o = c2w[:3, 3]
        rot = c2w[:3, :3]

        ys = np.arange(0, h, sample_step)
        xs = np.arange(0, w, sample_step)
        uu, vv = np.meshgrid(xs, ys)
        valid = mask[vv, uu] < 8
        if not np.any(valid):
            continue
        uu = uu[valid].astype(np.float64)
        vv = vv[valid].astype(np.float64)
        dirs_cv = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=1)
        dirs_world = dirs_cv @ rot.T
        denom = dirs_world @ plane_normal
        forward_depth = dirs_cv[:, 2]
        keep = np.abs(denom) > 1e-6
        t = ((plane_origin - cam_o) @ plane_normal) / np.where(np.abs(denom) > 1e-6, denom, 1.0)
        keep &= t > 0.2
        keep &= forward_depth > 0
        if not np.any(keep):
            continue
        pts_world = cam_o[None, :] + dirs_world[keep] * t[keep, None]
        pts_road = world_to_road(pts_world, frame)
        gx = np.floor((pts_road[:, 0] - extents["x_min"]) / resolution).astype(np.int64)
        gz = np.floor((pts_road[:, 2] - extents["z_min"]) / resolution).astype(np.int64)
        inside = (gx >= 0) & (gx < width_cells) & (gz >= 0) & (gz < height_cells)
        if not np.any(inside):
            continue
        gx = gx[inside]
        gz = gz[inside]
        colors = image[vv[keep][inside].astype(np.int64), uu[keep][inside].astype(np.int64)].astype(np.float64)
        weights = 1.0 / np.maximum(t[keep][inside], 1.0)
        np.add.at(rgb_sum, (gz, gx), colors * weights[:, None])
        np.add.at(weight_sum, (gz, gx), weights)
        np.add.at(hit_count, (gz, gx), 1)
        views_used += 1
        rays_used += int(len(gx))

    filled = weight_sum > 0
    texture = np.zeros((height_cells, width_cells, 3), dtype=np.uint8)
    texture[filled] = np.clip(rgb_sum[filled] / weight_sum[filled, None], 0, 255).astype(np.uint8)
    # Fill tiny holes for readability while preserving real sampled coverage.
    if np.any(filled):
        bgr = cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)
        missing = np.where(filled, 0, 255).astype(np.uint8)
        inpainted = cv2.inpaint(bgr, missing, 3, cv2.INPAINT_TELEA)
        texture = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        broad = cv2.dilate(filled.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
        texture[~broad] = 0
        texture = cv2.convertScaleAbs(texture, alpha=1.15, beta=8)

    return {
        "texture": texture,
        "filled": filled,
        "hit_count": hit_count,
        "views_used": views_used,
        "rays_used": rays_used,
        "coverage_ratio": float(np.count_nonzero(filled) / filled.size),
        "width_cells": width_cells,
        "height_cells": height_cells,
    }


def build_ground_mesh(texture_info: dict, extents: dict, resolution: float) -> trimesh.Trimesh:
    texture = texture_info["texture"]
    filled = cv2.dilate(texture_info["filled"].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    vertices = []
    faces = []
    colors = []
    h, w = filled.shape
    for gz in range(h):
        z = extents["z_min"] + gz * resolution
        for gx in range(w):
            if not filled[gz, gx]:
                continue
            x = extents["x_min"] + gx * resolution
            color = texture[gz, gx]
            idx = len(vertices)
            vertices.extend(
                [
                    [x, 0.0, z],
                    [x + resolution, 0.0, z],
                    [x + resolution, 0.0, z + resolution],
                    [x, 0.0, z + resolution],
                ]
            )
            faces.extend([[idx, idx + 2, idx + 1], [idx, idx + 3, idx + 2]])
            colors.extend([color.tolist() + [255]] * 4)
    if not vertices:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        vertex_colors=np.asarray(colors, dtype=np.uint8),
        process=False,
    )


def build_grid_mesh(texture: np.ndarray, filled: np.ndarray, extents: dict, resolution: float, stride: int = 1) -> trimesh.Trimesh:
    stride = max(1, int(stride))
    vertices = []
    faces = []
    colors = []
    h, w = filled.shape
    for gz in range(0, h, stride):
        for gx in range(0, w, stride):
            block = filled[gz : min(gz + stride, h), gx : min(gx + stride, w)]
            if not np.any(block):
                continue
            z = extents["z_min"] + gz * resolution
            x = extents["x_min"] + gx * resolution
            x1 = extents["x_min"] + min(gx + stride, w) * resolution
            z1 = extents["z_min"] + min(gz + stride, h) * resolution
            color_block = texture[gz : min(gz + stride, h), gx : min(gx + stride, w)]
            sample = color_block[block]
            color = np.median(sample, axis=0).astype(np.uint8) if len(sample) else np.array([28, 30, 32], dtype=np.uint8)
            idx = len(vertices)
            vertices.extend([[x, 0.0, z], [x1, 0.0, z], [x1, 0.0, z1], [x, 0.0, z1]])
            faces.extend([[idx, idx + 2, idx + 1], [idx, idx + 3, idx + 2]])
            colors.extend([color.tolist() + [255]] * 4)
    if not vertices:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        vertex_colors=np.asarray(colors, dtype=np.uint8),
        process=False,
    )


def infer_dataset_context(source: Path, args: argparse.Namespace) -> dict:
    context = {
        "dataset": args.dataset,
        "category": args.category,
        "scenario": args.scenario,
    }
    manifest_path = source / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        inputs = manifest.get("inputs", {})
        context["dataset"] = context["dataset"] or (Path(inputs["dataset_root"]) if inputs.get("dataset_root") else None)
        context["category"] = context["category"] or inputs.get("category")
        context["scenario"] = context["scenario"] or inputs.get("scenario")
    if context["dataset"] is None:
        context["dataset"] = Path("deepaccident_mini_dataset")
    return context


def load_bev_background(context: dict, args: argparse.Namespace, target_shape: tuple[int, int]) -> tuple[np.ndarray | None, dict]:
    if not args.use_bev_background:
        return None, {"status": "disabled"}
    if not context.get("category") or not context.get("scenario"):
        return None, {"status": "missing_context"}
    scenario = context["scenario"]
    folder = Path(context["dataset"]) / context["category"] / args.bev_agent / "BEV_instance_camera" / scenario
    if not folder.exists():
        return None, {"status": "missing_folder", "path": str(folder)}
    requested = folder / f"{scenario}_{args.bev_frame:03d}.npz"
    if requested.exists():
        chosen = requested
    else:
        candidates = sorted(folder.glob("*.npz"))
        if not candidates:
            return None, {"status": "missing_files", "path": str(folder)}
        chosen = min(
            candidates,
            key=lambda p: abs(int(p.stem.rsplit("_", 1)[-1]) - args.bev_frame),
        )
    with np.load(chosen) as data:
        image = np.asarray(data["data"], dtype=np.uint8)
    rotation = int(args.bev_rotation) % 4
    if rotation:
        image = np.rot90(image, rotation)
    if args.bev_flip_x:
        image = np.fliplr(image)
    if args.bev_flip_y:
        image = np.flipud(image)
    target_h, target_w = target_shape
    resized = np.asarray(Image.fromarray(image).resize((target_w, target_h), Image.Resampling.BILINEAR))
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 0.62
    hsv[..., 2] = np.clip(hsv[..., 2] * 0.82 + 42.0, 0, 255)
    muted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return muted, {
        "status": "ok",
        "source": str(chosen),
        "agent": args.bev_agent,
        "frame": int(chosen.stem.rsplit("_", 1)[-1]),
        "rotation": rotation,
        "flip_x": bool(args.bev_flip_x),
        "flip_y": bool(args.bev_flip_y),
        "note": "BEV_instance_camera is used as a camera-BEV background prior only; no lidar01 geometry is read.",
    }


def compose_display_texture(texture_info: dict, bev_texture: np.ndarray | None) -> dict:
    projected = texture_info["texture"]
    filled = texture_info["filled"].astype(bool)
    display = cv2.medianBlur(projected, 5)
    if bev_texture is not None:
        confidence = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 2.2)
        confidence = np.clip(confidence[..., None] * 1.7, 0.0, 1.0)
        display = np.clip(display.astype(np.float32) * confidence + bev_texture.astype(np.float32) * (1.0 - confidence), 0, 255).astype(np.uint8)
        display_filled = np.ones(filled.shape, dtype=bool)
    else:
        display_filled = cv2.dilate(filled.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    display = cv2.bilateralFilter(display, 5, 30, 30)
    return {"texture": display, "filled": display_filled}


def clean_mask(mask: np.ndarray, open_size: int = 3, close_size: int = 9) -> np.ndarray:
    out = mask.astype(np.uint8)
    if open_size > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    if close_size > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    return out.astype(bool)


def semantic_masks_from_texture(texture: np.ndarray, valid: np.ndarray, bev_texture: np.ndarray | None = None) -> dict:
    smooth = cv2.medianBlur(texture, 5)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]
    red = smooth[..., 0].astype(np.int16)
    green = smooth[..., 1].astype(np.int16)
    blue = smooth[..., 2].astype(np.int16)
    vegetation = (
        valid
        & (green > red + 14)
        & (green > blue + 8)
        & (hue >= 32)
        & (hue <= 94)
        & (val > 38)
    )
    roof_like = (
        valid
        & (sat > 42)
        & (val > 36)
        & (red > green + 16)
        & (red > blue + 8)
    )
    dark_constructed = valid & (sat > 48) & (val < 112) & ~vegetation
    building = roof_like | dark_constructed
    if bev_texture is not None:
        bev = cv2.medianBlur(bev_texture, 5)
        br = bev[..., 0].astype(np.int16)
        bg = bev[..., 1].astype(np.int16)
        bb = bev[..., 2].astype(np.int16)
        bhsv = cv2.cvtColor(bev, cv2.COLOR_RGB2HSV)
        bval = bhsv[..., 2]
        bsat = bhsv[..., 1]
        bev_tree = valid & (bg > 118) & (bg > br + 24) & (bg > bb + 20)
        bev_building = valid & (br > 48) & (br < 112) & (bg < 68) & (bb < 68) & (bsat > 45) & (bval < 145)
        vegetation |= bev_tree
        building |= bev_building
    vegetation = clean_mask(vegetation, open_size=5, close_size=11)
    building = clean_mask(building & ~cv2.dilate(vegetation.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool), open_size=5, close_size=13)
    road = clean_mask(valid & ~vegetation & ~building & (sat < 105) & (val > 42), open_size=3, close_size=7)
    return {"building": building, "vegetation": vegetation, "road": road}


def component_records(mask: np.ndarray, resolution: float, min_area_m2: float, max_components: int) -> list[dict]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    records = []
    h, w = mask.shape
    for label_id in range(1, count):
        x, y, bw, bh, area = stats[label_id]
        area_m2 = float(area) * resolution * resolution
        if area_m2 < min_area_m2:
            continue
        width_m = bw * resolution
        depth_m = bh * resolution
        if width_m < 1.4 or depth_m < 1.4:
            continue
        aspect = max(width_m / max(depth_m, 1e-6), depth_m / max(width_m, 1e-6))
        fill_ratio = area / max(bw * bh, 1)
        touches_border = x == 0 or y == 0 or x + bw >= w - 1 or y + bh >= h - 1
        records.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(bw),
                "h": int(bh),
                "area": int(area),
                "area_m2": area_m2,
                "centroid": centroids[label_id].tolist(),
                "aspect": float(aspect),
                "fill_ratio": float(fill_ratio),
                "touches_border": bool(touches_border),
            }
        )
    records.sort(key=lambda item: item["area_m2"], reverse=True)
    return records[:max_components]


def append_box(
    vertices: list,
    faces: list,
    colors: list,
    xmin: float,
    xmax: float,
    zmin: float,
    zmax: float,
    height: float,
    color: np.ndarray,
    include_roof: bool = True,
) -> None:
    bottom = 0.04
    color = np.clip(color.astype(np.float32), 35, 245).astype(np.uint8)
    side = np.clip(color.astype(np.float32) * 0.72, 20, 210).astype(np.uint8)
    roof = np.clip(color.astype(np.float32) * 1.08 + 8, 30, 255).astype(np.uint8)
    quads = [
        ([[xmin, bottom, zmin], [xmin, height, zmin], [xmin, height, zmax], [xmin, bottom, zmax]], side),
        ([[xmax, bottom, zmin], [xmax, bottom, zmax], [xmax, height, zmax], [xmax, height, zmin]], side),
        ([[xmin, bottom, zmin], [xmax, bottom, zmin], [xmax, height, zmin], [xmin, height, zmin]], side),
        ([[xmin, bottom, zmax], [xmin, height, zmax], [xmax, height, zmax], [xmax, bottom, zmax]], side),
    ]
    if include_roof:
        quads.insert(0, ([[xmin, height, zmin], [xmax, height, zmin], [xmax, height, zmax], [xmin, height, zmax]], roof))
    for quad, quad_color in quads:
        idx = len(vertices)
        vertices.extend(quad)
        faces.extend([[idx, idx + 1, idx + 2], [idx, idx + 2, idx + 3]])
        colors.extend([quad_color.tolist() + [255]] * 4)


def build_building_mesh(mask: np.ndarray, texture: np.ndarray, extents: dict, resolution: float, args: argparse.Namespace) -> tuple[trimesh.Trimesh, dict]:
    records = component_records(mask, resolution, args.building_min_area_m2, args.max_buildings * 3)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    colors: list[list[int]] = []
    used = []
    rng = np.random.default_rng(args.seed)
    for record in records:
        if len(used) >= args.max_buildings:
            break
        width_m = record["w"] * resolution
        depth_m = record["h"] * resolution
        if record["aspect"] > 7.5 and record["fill_ratio"] < 0.72:
            continue
        if record["touches_border"] and record["aspect"] > 4.5:
            continue
        if record["area_m2"] > 1800 and record["fill_ratio"] < 0.55:
            continue
        x0 = extents["x_min"] + record["x"] * resolution
        x1 = extents["x_min"] + (record["x"] + record["w"]) * resolution
        z0 = extents["z_min"] + record["y"] * resolution
        z1 = extents["z_min"] + (record["y"] + record["h"]) * resolution
        pad = min(0.35, max(width_m, depth_m) * 0.03)
        patch = texture[record["y"] : record["y"] + record["h"], record["x"] : record["x"] + record["w"]]
        color = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
        height = float(np.clip(2.4 + math.sqrt(record["area_m2"]) * 0.18 + rng.uniform(-0.4, 0.9), 2.8, 8.5))
        append_box(vertices, faces, colors, x0 + pad, x1 - pad, z0 + pad, z1 - pad, height, color, include_roof=False)
        used.append({**record, "height_m": height})
    if not vertices:
        mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
    else:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int64),
            vertex_colors=np.asarray(colors, dtype=np.uint8),
            process=False,
        )
    return mesh, {
        "components_considered": len(records),
        "buildings_exported": len(used),
        "min_area_m2": args.building_min_area_m2,
    }


def append_billboard_tree(vertices: list, faces: list, colors: list, x: float, z: float, radius: float, height: float, color: np.ndarray) -> None:
    trunk_color = np.array([88, 64, 39], dtype=np.uint8)
    canopy_color = np.clip(color.astype(np.float32) * np.array([0.72, 1.08, 0.66]) + np.array([8, 18, 4]), 25, 245).astype(np.uint8)
    trunk_w = max(0.16, radius * 0.12)
    append_box(vertices, faces, colors, x - trunk_w, x + trunk_w, z - trunk_w, z + trunk_w, height * 0.48, trunk_color)
    y0 = height * 0.38
    y1 = height
    quads = [
        [[x - radius, y0, z], [x + radius, y0, z], [x + radius, y1, z], [x - radius, y1, z]],
        [[x, y0, z - radius], [x, y0, z + radius], [x, y1, z + radius], [x, y1, z - radius]],
    ]
    for quad in quads:
        idx = len(vertices)
        vertices.extend(quad)
        faces.extend([[idx, idx + 1, idx + 2], [idx, idx + 2, idx + 3]])
        colors.extend([canopy_color.tolist() + [235]] * 4)


def build_tree_mesh(mask: np.ndarray, texture: np.ndarray, extents: dict, resolution: float, args: argparse.Namespace) -> tuple[trimesh.Trimesh, dict]:
    records = component_records(mask, resolution, min_area_m2=3.0, max_components=args.max_trees * 4)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    colors: list[list[int]] = []
    used = []
    rng = np.random.default_rng(args.seed + 11)
    for record in records:
        if len(used) >= args.max_trees:
            break
        width_m = record["w"] * resolution
        depth_m = record["h"] * resolution
        if max(width_m, depth_m) > 12.0 or record["aspect"] > 3.2:
            continue
        cx_px, cy_px = record["centroid"]
        x = extents["x_min"] + float(cx_px) * resolution
        z = extents["z_min"] + float(cy_px) * resolution
        radius = float(np.clip(math.sqrt(record["area_m2"] / math.pi) * 0.72, 0.8, 3.8))
        height = float(np.clip(radius * 1.55 + rng.uniform(1.2, 2.3), 2.6, 7.2))
        patch = texture[record["y"] : record["y"] + record["h"], record["x"] : record["x"] + record["w"]]
        color = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
        append_billboard_tree(vertices, faces, colors, x, z, radius, height, color)
        used.append({**record, "height_m": height, "radius_m": radius})
    if not vertices:
        mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
    else:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int64),
            vertex_colors=np.asarray(colors, dtype=np.uint8),
            process=False,
        )
    return mesh, {"components_considered": len(records), "trees_exported": len(used)}


def write_mask_preview(path: Path, masks: dict) -> None:
    building = masks["building"]
    vegetation = masks["vegetation"]
    road = masks["road"]
    preview = np.zeros((*building.shape, 3), dtype=np.uint8)
    preview[road] = [92, 100, 108]
    preview[vegetation] = [44, 168, 76]
    preview[building] = [190, 72, 64]
    Image.fromarray(preview).save(path)


def read_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)
    colors = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype(np.uint8)
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
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def filter_vertical_cloud(source: Path, frame: dict, extents: dict, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    points_world, colors = read_ply_points(source / "reconstruction" / "points_world.ply")
    points_road = world_to_road(points_world, frame)
    keep = (
        (points_road[:, 0] >= extents["x_min"])
        & (points_road[:, 0] <= extents["x_max"])
        & (points_road[:, 2] >= extents["z_min"])
        & (points_road[:, 2] <= extents["z_max"])
        & (points_road[:, 1] >= args.height_min)
        & (points_road[:, 1] <= args.height_max)
    )
    points_road = points_road[keep]
    colors = colors[keep]
    before = len(points_road)
    if args.max_vertical_points > 0 and len(points_road) > args.max_vertical_points:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(points_road), size=args.max_vertical_points, replace=False))
        points_road = points_road[idx]
        colors = colors[idx]
    report = {
        "input_points": int(len(points_world)),
        "vertical_points_before_sampling": int(before),
        "vertical_points_exported": int(len(points_road)),
        "height_range_m": [args.height_min, args.height_max],
    }
    return points_road.astype(np.float32), colors, report


def transform_tracks(agent_tracks: dict, diagnostics: dict, frame: dict) -> dict:
    transformed = {
        "schema_version": "0.2",
        "coordinate_system": "road frame: x=right, y=road-normal up, z=ego forward. This corrects the previous viewer orientation.",
        "origin_world": frame["origin_world"].tolist(),
        "right_world": frame["right_world"].tolist(),
        "up_world": frame["up_world"].tolist(),
        "forward_world": frame["forward_world"].tolist(),
        "agents": {},
    }
    for agent_name, agent in agent_tracks["agents"].items():
        samples = []
        for sample in agent["samples"]:
            pos = np.asarray(sample["position_world"], dtype=np.float64)
            road = world_to_road(pos[None, :], frame)[0]
            yaw = float(math.atan2(np.dot(frame["right_world"], [math.cos(sample["yaw_rad"]), math.sin(sample["yaw_rad"]), 0.0]), 1.0))
            item = dict(sample)
            item["position_viewer"] = [float(road[0]), float(road[1]), float(road[2])]
            item["road_frame_note"] = "viewer position was recomputed after plane/orientation correction"
            item["yaw_rad"] = yaw
            samples.append(item)
        transformed["agents"][agent_name] = {
            "color": agent.get("color", AGENT_COLORS.get(agent_name, "#ffffff")),
            "dimensions": agent["dimensions"],
            "samples": samples,
        }
    transformed["closest_approach"] = diagnostics.get("closest_approach")
    return transformed


def write_viewer(out: Path, extents: dict) -> None:
    viewer = out / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    center_x = (extents["x_min"] + extents["x_max"]) / 2.0
    center_z = (extents["z_min"] + extents["z_max"]) / 2.0
    span = max(extents["x_max"] - extents["x_min"], extents["z_max"] - extents["z_min"])
    camera_y = max(45, span * 0.55)
    camera_z = extents["z_min"] - max(35, span * 0.25)
    (viewer / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Readable camera-only reconstruction</title>
  <style>
    html, body {{ margin:0; height:100%; overflow:hidden; background:#07090c; color:#e8eaed; font:13px system-ui, sans-serif; }}
    #hud {{ position:absolute; left:12px; top:12px; z-index:2; width:min(440px, calc(100vw - 24px)); background:rgba(7,9,12,.78); border:1px solid #2d333b; padding:10px 12px; box-sizing:border-box; }}
    #hud strong {{ display:block; margin-bottom:4px; font-size:14px; }}
    #hud .row {{ display:flex; justify-content:space-between; gap:12px; white-space:nowrap; }}
    #hud input {{ width:100%; margin-top:8px; }}
    #legend {{ display:grid; grid-template-columns:1fr 1fr; gap:4px 10px; margin-top:8px; }}
    .swatch {{ display:inline-block; width:10px; height:10px; margin-right:6px; border-radius:50%; vertical-align:-1px; }}
    #buttons {{ display:flex; gap:6px; margin-top:8px; }}
    button {{ background:#1f2937; color:#e8eaed; border:1px solid #374151; padding:5px 8px; cursor:pointer; }}
    canvas {{ display:block; }}
  </style>
  <script type="importmap">{{"imports":{{"three":"../../../viewer/vendor/three/build/three.module.js","three/addons/":"../../../viewer/vendor/three/examples/jsm/"}}}}</script>
</head>
<body>
  <div id="hud">
    <strong>Readable road-frame reconstruction</strong>
    <div class="row"><span id="status">loading</span><span id="frameLabel"></span></div>
    <input id="frame" type="range" min="0" max="0" value="0" step="1" />
    <div id="closest"></div>
    <div id="buttons"><button id="front">front view</button><button id="top">top view</button><button id="flip">flip 180</button><button id="points">VGGT points off</button></div>
    <div id="legend"></div>
  </div>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
    import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07090c);
    const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.05, 5000);
    const renderer = new THREE.WebGLRenderer({{ antialias:true, preserveDrawingBuffer:true }});
    renderer.setSize(innerWidth, innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    document.body.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const sceneRoot = new THREE.Group();
    scene.add(sceneRoot);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x20242b, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(-30, 80, -40);
    scene.add(sun);

    const target = new THREE.Vector3({center_x:.3f}, 0, {center_z:.3f});
    function setFrontView() {{
      sceneRoot.rotation.y = 0;
      camera.position.set({center_x:.3f}, {camera_y:.3f}, {camera_z:.3f});
      controls.target.copy(target);
      camera.lookAt(target);
    }}
    function setTopView() {{
      camera.position.set({center_x:.3f}, {max(camera_y * 2.15, 115):.3f}, {center_z:.3f} + .001);
      controls.target.copy(target);
      camera.lookAt(target);
    }}
    setTopView();

    const statusEl = document.getElementById('status');
    const frameEl = document.getElementById('frame');
    const frameLabelEl = document.getElementById('frameLabel');
    const closestEl = document.getElementById('closest');
    const legendEl = document.getElementById('legend');
    const markerGroup = new THREE.Group();
    sceneRoot.add(markerGroup);
    let tracks = null;
    let frames = [];
    let markers = [];
    let pointObjects = [];
    let pointsVisible = false;

    function hexToNumber(hex) {{ return Number.parseInt(hex.replace('#', ''), 16); }}
    function makeLine(points, color) {{
      const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p[0], p[1] + .25, p[2])));
      return new THREE.Line(geometry, new THREE.LineBasicMaterial({{ color, linewidth:2 }}));
    }}
    function addTracks() {{
      legendEl.innerHTML = '';
      for (const [agent, data] of Object.entries(tracks.agents)) {{
        const color = hexToNumber(data.color || '#ffffff');
        sceneRoot.add(makeLine(data.samples.map(s => s.position_viewer), color));
        const div = document.createElement('div');
        div.innerHTML = `<span class="swatch" style="background:${{data.color}}"></span>${{agent}}`;
        legendEl.appendChild(div);
        const marker = new THREE.Mesh(
          new THREE.BoxGeometry(Math.max(.7, data.dimensions.width_m), Math.max(.7, data.dimensions.height_m), Math.max(1, data.dimensions.length_m)),
          new THREE.MeshBasicMaterial({{ color, wireframe:true }})
        );
        marker.userData.agent = agent;
        markerGroup.add(marker);
        markers.push(marker);
      }}
      frames = [...new Set(Object.values(tracks.agents).flatMap(d => d.samples.map(s => s.frame)))].sort((a, b) => a - b);
      frameEl.max = Math.max(0, frames.length - 1);
      const c = tracks.closest_approach;
      const startIdx = c ? Math.max(0, frames.indexOf(c.frame)) : 0;
      frameEl.value = startIdx;
      updateFrame(startIdx);
    }}
    function updateFrame(idx) {{
      if (!tracks || frames.length === 0) return;
      const frame = frames[idx];
      frameLabelEl.textContent = `frame ${{frame}}`;
      markers.forEach(marker => {{
        const sample = tracks.agents[marker.userData.agent].samples.find(s => s.frame === frame);
        if (!sample) {{ marker.visible = false; return; }}
        marker.visible = true;
        marker.position.set(sample.position_viewer[0], sample.position_viewer[1] + .8, sample.position_viewer[2]);
        marker.rotation.y = -sample.yaw_rad;
      }});
    }}
    frameEl.addEventListener('input', () => updateFrame(Number(frameEl.value)));
    document.getElementById('front').onclick = setFrontView;
    document.getElementById('top').onclick = setTopView;
    document.getElementById('flip').onclick = () => {{ sceneRoot.rotation.y += Math.PI; }};
    document.getElementById('points').onclick = () => {{
      pointsVisible = !pointsVisible;
      pointObjects.forEach(obj => obj.visible = pointsVisible);
      document.getElementById('points').textContent = pointsVisible ? 'VGGT points on' : 'VGGT points off';
    }};

    new GLTFLoader().load('../scene.glb', gltf => {{
      gltf.scene.traverse(obj => {{
        if (obj.isMesh) {{
          obj.material = new THREE.MeshBasicMaterial({{ vertexColors:true, side:THREE.DoubleSide }});
        }}
        if (obj.isPoints) {{
          obj.visible = false;
          if (obj.material) obj.material.size = 0.035;
          pointObjects.push(obj);
        }}
      }});
      sceneRoot.add(gltf.scene);
      statusEl.textContent = 'single-plane RGB mesh loaded';
    }}, undefined, err => {{
      console.error(err);
      statusEl.textContent = 'scene.glb failed';
    }});
    fetch('../replay/agent_tracks_road_frame.json').then(r => r.json()).then(json => {{
      tracks = json;
      addTracks();
      const c = tracks.closest_approach;
      if (c) closestEl.textContent = `closest proxy: ${{c.agents.join(' / ')}} frame ${{c.frame}}, clearance ${{c.proxy_clearance_xy_m.toFixed(2)}}m`;
    }}).catch(err => {{
      console.error(err);
      closestEl.textContent = 'track JSON failed';
    }});
    addEventListener('resize', () => {{
      camera.aspect = innerWidth / innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(innerWidth, innerHeight);
    }});
    function animate() {{
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }}
    animate();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_preview(path: Path, texture_info: dict, vertical_points: np.ndarray, vertical_colors: np.ndarray, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    texture = texture_info["texture"]
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    ax.imshow(np.flipud(texture), interpolation="nearest")
    if len(vertical_points):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(vertical_points), min(30000, len(vertical_points)), replace=False)
        pts = vertical_points[idx]
        ax.scatter(pts[:, 0], texture.shape[0] - pts[:, 2], s=0.1, c=vertical_colors[idx] / 255.0, alpha=0.5)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    for rel in ["reconstruction", "replay", "viewer", "reports"]:
        (out / rel).mkdir(parents=True, exist_ok=True)

    agent_tracks = load_json(source / "replay" / "agent_tracks.json")
    diagnostics = load_json(source / "replay" / "accident_diagnostics.json")
    cameras = load_json(source / "reconstruction" / "cameras.json")
    dataset_context = infer_dataset_context(source, args)
    road_frame = fit_road_frame(agent_tracks, diagnostics)
    extents = trajectory_extents(agent_tracks, road_frame, args.trajectory_margin)

    texture_info = rasterize_ground_texture(source, cameras, road_frame, extents, args.grid_resolution, args.sample_step)
    bev_texture, bev_report = load_bev_background(dataset_context, args, texture_info["texture"].shape[:2])
    display_texture_info = compose_display_texture(texture_info, bev_texture)
    ground_mesh = build_grid_mesh(
        display_texture_info["texture"],
        display_texture_info["filled"],
        extents,
        args.grid_resolution,
        stride=args.background_stride,
    )
    semantic_masks = semantic_masks_from_texture(
        display_texture_info["texture"],
        display_texture_info["filled"],
        bev_texture=bev_texture,
    )
    building_mesh, building_report = build_building_mesh(
        semantic_masks["building"],
        display_texture_info["texture"],
        extents,
        args.grid_resolution,
        args,
    )
    tree_mesh, tree_report = build_tree_mesh(
        semantic_masks["vegetation"],
        display_texture_info["texture"],
        extents,
        args.grid_resolution,
        args,
    )
    vertical_points, vertical_colors, vertical_report = filter_vertical_cloud(source, road_frame, extents, args)
    write_ply(out / "reconstruction" / "vertical_points_road_frame.ply", vertical_points, vertical_colors)
    vertical_cloud = trimesh.PointCloud(vertices=vertical_points.astype(np.float32), colors=vertical_colors)
    scene = trimesh.Scene()
    if len(ground_mesh.vertices):
        scene.add_geometry(ground_mesh, node_name="single_plane_rgb_bev_ground")
    if len(building_mesh.vertices):
        scene.add_geometry(building_mesh, node_name="extruded_rgb_building_background")
    if len(tree_mesh.vertices):
        scene.add_geometry(tree_mesh, node_name="rgb_tree_and_vegetation_background")
    if args.include_vggt_points and len(vertical_cloud.vertices):
        scene.add_geometry(vertical_cloud, node_name="cropped_vertical_vggt_points")
    scene.export(out / "scene.glb")

    Image.fromarray(texture_info["texture"]).save(out / "reports" / "ground_texture_projected_rgb_only.png")
    Image.fromarray(display_texture_info["texture"]).save(out / "reports" / "ground_texture.png")
    if bev_texture is not None:
        Image.fromarray(bev_texture).save(out / "reports" / "camera_bev_background_prior.png")
    write_mask_preview(out / "reports" / "background_mask_preview.png", semantic_masks)
    preview_points = vertical_points if args.include_vggt_points else np.zeros((0, 3), dtype=np.float32)
    preview_colors = vertical_colors if args.include_vggt_points else np.zeros((0, 3), dtype=np.uint8)
    write_preview(out / "reports" / "readable_topdown_preview.png", display_texture_info, preview_points, preview_colors, args.seed)
    transformed_tracks = transform_tracks(agent_tracks, diagnostics, road_frame)
    write_json(out / "replay" / "agent_tracks_road_frame.json", transformed_tracks)
    shutil.copy2(source / "replay" / "accident_diagnostics.json", out / "replay" / "accident_diagnostics.json")
    write_viewer(out, extents)

    report = {
        "schema_version": "0.1",
        "source": str(source),
        "status": "ok",
        "lidar_used": False,
        "corrections": [
            "Fitted road plane from calibration-derived vehicle trajectories.",
            "Recomputed viewer frame so +Z follows ego_vehicle motion direction.",
            "Projected masked RGB camera frames onto the road plane for readable ground texture.",
            "Composed one single ground plane from camera projection plus optional BEV_instance_camera camera background prior.",
            "Replaced sparse layered VGGT vertical display with RGB-derived extruded building/tree background meshes.",
            "Cropped vertical dense points are exported for audit only unless --include-vggt-points is enabled.",
        ],
        "dataset_context": {
            "dataset": str(dataset_context["dataset"]) if dataset_context.get("dataset") else None,
            "category": dataset_context.get("category"),
            "scenario": dataset_context.get("scenario"),
        },
        "road_frame": {
            "origin_world": road_frame["origin_world"].tolist(),
            "right_world": road_frame["right_world"].tolist(),
            "up_world": road_frame["up_world"].tolist(),
            "forward_world": road_frame["forward_world"].tolist(),
        },
        "extents_road_frame": extents,
        "ground_texture": {
            "resolution_m": args.grid_resolution,
            "sample_step_px": args.sample_step,
            "views_used": texture_info["views_used"],
            "rays_used": texture_info["rays_used"],
            "coverage_ratio": texture_info["coverage_ratio"],
            "width_cells": texture_info["width_cells"],
            "height_cells": texture_info["height_cells"],
            "display_cells": int(np.count_nonzero(display_texture_info["filled"])),
        },
        "bev_background": bev_report,
        "mesh_background": {
            "vggt_points_in_scene_glb": bool(args.include_vggt_points),
            "building_mesh": building_report,
            "tree_mesh": tree_report,
            "single_ground_plane": True,
        },
        "vertical_cloud": vertical_report,
        "outputs": {
            "viewer": "viewer/index.html",
            "scene_glb": "scene.glb",
            "ground_texture": "reports/ground_texture.png",
            "projected_rgb_ground_texture": "reports/ground_texture_projected_rgb_only.png",
            "background_mask_preview": "reports/background_mask_preview.png",
            "topdown_preview": "reports/readable_topdown_preview.png",
            "no_lidar_audit": "reports/no_lidar_audit.md",
            "tracks": "replay/agent_tracks_road_frame.json",
        },
    }
    write_json(out / "manifest.json", report)
    (out / "reports" / "readable_quality_report.md").write_text(
        f"""# Readable Camera-only Scene

Status: `ok`

This output replaces the previous raw dense-cloud viewer with a road-frame
visualization intended to be readable as a road/city scene.

## Corrections

- Road plane fitted from calibration-derived vehicle trajectories.
- Viewer frame corrected so `+Z` follows `ego_vehicle` motion.
- Masked RGB camera frames projected onto the road plane.
- One single display ground plane is composed for readability.
- RGB/BEV color masks are converted into extruded building/tree background meshes.
- VGGT dense vertical points are exported for audit but not included in `scene.glb` by default.
- LiDAR geometry used: `false`.

## Metrics

- Ground texture coverage: `{texture_info['coverage_ratio']:.3f}`
- RGB projection views used: `{texture_info['views_used']}`
- RGB projection samples used: `{texture_info['rays_used']}`
- Vertical points exported: `{vertical_report['vertical_points_exported']}`
- Buildings exported: `{building_report['buildings_exported']}`
- Trees/vegetation proxies exported: `{tree_report['trees_exported']}`
- BEV background prior: `{bev_report['status']}`

## Main Output

- `viewer/index.html`
- `scene.glb`
- `reports/ground_texture.png`
- `reports/background_mask_preview.png`
- `reports/readable_topdown_preview.png`
""",
        encoding="utf-8",
    )
    (out / "reports" / "no_lidar_audit.md").write_text(
        f"""# No-LiDAR Audit

Status: `ok`

- LiDAR geometry used: `false`
- `lidar01` point files read: `false`
- Legacy LiDAR PLY/GLB assets read: `false`
- Calibration matrices with `lidar_to_*` names used only as camera extrinsics: `true`
- RGB camera views projected: `{texture_info['views_used']}`
- Camera-BEV background prior: `{bev_report['status']}`
- VGGT points included in `scene.glb`: `{str(bool(args.include_vggt_points)).lower()}`

The exported `scene.glb` is built from one RGB/BEV display ground mesh plus
RGB-derived building/tree proxy meshes. Cropped VGGT vertical points are
written to `reconstruction/vertical_points_road_frame.ply` for audit only and
are not included in the viewer unless `--include-vggt-points` is set.
""",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
