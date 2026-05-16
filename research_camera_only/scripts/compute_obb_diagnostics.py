#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def axes(yaw: float) -> tuple[tuple[float, float], tuple[float, float]]:
    forward = (math.cos(yaw), math.sin(yaw))
    right = (-math.sin(yaw), math.cos(yaw))
    return forward, right


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm(a: tuple[float, float]) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: tuple[float, float]) -> tuple[float, float]:
    n = max(norm(a), 1e-12)
    return (a[0] / n, a[1] / n)


def obb_gap(a: dict, b: dict, dims_a: dict, dims_b: dict) -> tuple[float, float, bool]:
    ca = (float(a["x_m"]), float(a["y_m"]))
    cb = (float(b["x_m"]), float(b["y_m"]))
    axes_a = axes(float(a["yaw_rad"]))
    axes_b = axes(float(b["yaw_rad"]))
    test_axes = [normalize(axis) for axis in (*axes_a, *axes_b)]
    half_a = [float(dims_a["length"]) / 2.0, float(dims_a["width"]) / 2.0]
    half_b = [float(dims_b["length"]) / 2.0, float(dims_b["width"]) / 2.0]
    delta = (cb[0] - ca[0], cb[1] - ca[1])
    max_gap = -1e9
    overlap = True
    for axis in test_axes:
        center_distance = abs(dot(delta, axis))
        radius_a = sum(abs(dot(ax, axis)) * h for ax, h in zip(axes_a, half_a))
        radius_b = sum(abs(dot(ax, axis)) * h for ax, h in zip(axes_b, half_b))
        gap = center_distance - (radius_a + radius_b)
        max_gap = max(max_gap, gap)
        if gap > 0:
            overlap = False
    center = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1])
    return max_gap, center, overlap


def pose_by_frame(vehicle: dict) -> dict[int, dict]:
    return {int(pose["frame_index"]): pose for pose in vehicle["poses"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute 2D OBB gap diagnostics from proxy tracks.")
    parser.add_argument("--tracks", required=True, help="vehicle_tracks.json path.")
    parser.add_argument("--vehicle-a", required=True)
    parser.add_argument("--vehicle-b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale-alignment-path", default=None)
    parser.add_argument("--uncertainty-note", default="Pose and scale uncertainty must be reported with accident claims.")
    args = parser.parse_args()

    data = json.loads(Path(args.tracks).read_text())
    vehicles = {vehicle["vehicle_id"]: vehicle for vehicle in data["vehicles"]}
    if args.vehicle_a not in vehicles or args.vehicle_b not in vehicles:
        raise SystemExit("requested vehicle IDs not found in tracks")

    va = vehicles[args.vehicle_a]
    vb = vehicles[args.vehicle_b]
    pa = pose_by_frame(va)
    pb = pose_by_frame(vb)
    common = sorted(set(pa).intersection(pb))
    if not common:
        raise SystemExit("no common frame indices")

    frames = []
    best = None
    for frame_index in common:
        gap, center_distance, overlap = obb_gap(pa[frame_index], pb[frame_index], va["dimensions_m"], vb["dimensions_m"])
        row = {
            "frame_index": frame_index,
            "obb_gap_m": gap,
            "center_distance_m": center_distance,
            "overlap": overlap,
        }
        frames.append(row)
        if best is None or gap < best["obb_gap_m"]:
            best = row

    result = {
        "schema_version": "0.1",
        "case_id": data.get("case_id", "unknown_case"),
        "vehicle_pair": [args.vehicle_a, args.vehicle_b],
        "frames": frames,
        "best_frame": best,
        "scale_alignment_path": args.scale_alignment_path,
        "uncertainty_note": args.uncertainty_note,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(out), "frames": len(frames), "best_frame": best}, indent=2))


if __name__ == "__main__":
    main()
