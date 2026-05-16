#!/usr/bin/env python3
from __future__ import annotations

import argparse
import binascii
import json
import os
import shutil
import struct
import zlib
from pathlib import Path


DEFAULT_AGENTS = [
    "ego_vehicle",
    "ego_vehicle_behind",
    "other_vehicle",
    "other_vehicle_behind",
]

DYNAMIC_CLASSES = ["car", "truck", "bus", "van", "motorcycle", "bicycle", "pedestrian", "person"]


def detect_source_kind(source: Path, agents: list[str], category: str, scenario: str, camera: str) -> str:
    if all((source / agent / "frame_0000.jpg").exists() for agent in agents):
        return "viewer_frames"
    deep_paths = [
        source / category / agent / camera / scenario
        for agent in agents
    ]
    if all(path.exists() for path in deep_paths):
        return "deepaccident_rgb"
    raise FileNotFoundError(
        "Could not detect source kind. Expected viewer_frames/<agent>/frame_0000.jpg "
        "or DeepAccident <root>/<category>/<agent>/<camera>/<scenario>/."
    )


def frame_source_path(
    source: Path,
    kind: str,
    agent: str,
    frame_index: int,
    category: str,
    scenario: str,
    camera: str,
) -> Path:
    if kind == "viewer_frames":
        return source / agent / f"frame_{frame_index:04d}.jpg"
    if kind == "deepaccident_rgb":
        source_frame = frame_index + 1
        return source / category / agent / camera / scenario / f"{scenario}_{source_frame:03d}.jpg"
    raise ValueError(f"unsupported source kind: {kind}")


def place_file(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
        dst.symlink_to(rel)


def jpeg_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8"):
        raise ValueError(f"expected JPEG input for placeholder mask size: {path}")
    idx = 2
    while idx < len(data):
        while idx < len(data) and data[idx] == 0xFF:
            idx += 1
        if idx >= len(data):
            break
        marker = data[idx]
        idx += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if idx + 2 > len(data):
            break
        size = int.from_bytes(data[idx:idx + 2], "big")
        if size < 2 or idx + size > len(data):
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if size < 7:
                break
            height = int.from_bytes(data[idx + 3:idx + 5], "big")
            width = int.from_bytes(data[idx + 5:idx + 7], "big")
            return width, height
        idx += size
    raise ValueError(f"could not read JPEG dimensions: {path}")


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_grayscale_png(path: Path, width: int, height: int, value: int = 0) -> None:
    raw_row = bytes([0]) + bytes([value]) * width
    payload = raw_row * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(payload, level=9))
        + png_chunk(b"IEND", b"")
    )


def write_empty_mask(rgb_path: Path, mask_path: Path) -> None:
    width, height = jpeg_size(rgb_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    write_grayscale_png(mask_path, width, height)


def build_manifest(args: argparse.Namespace) -> dict:
    source = Path(args.source).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    agents = args.agents or DEFAULT_AGENTS
    kind = args.source_kind
    if kind == "auto":
        kind = detect_source_kind(source, agents, args.category, args.scenario, args.camera)

    rows = []
    for frame_index in range(args.frame_start, args.frame_end + 1):
        for agent in agents:
            src = frame_source_path(source, kind, agent, frame_index, args.category, args.scenario, args.camera)
            if not src.exists():
                raise FileNotFoundError(src)
            rgb_rel = Path("rgb_frames") / agent / f"frame_{frame_index:04d}.jpg"
            rgb_dst = out / rgb_rel
            place_file(src, rgb_dst, args.copy)

            mask_rel: Path | None = None
            mask_status = "not_run"
            qa_status = "pending_segmentation"
            if args.make_empty_masks:
                mask_rel = Path("masks") / agent / f"frame_{frame_index:04d}.png"
                write_empty_mask(rgb_dst, out / mask_rel)
                mask_status = "placeholder_empty"
                qa_status = "not_evidence"

            rows.append(
                {
                    "agent": agent,
                    "frame_index": frame_index,
                    "source_frame": src.name,
                    "rgb_path": str(rgb_rel),
                    "mask_path": str(mask_rel) if mask_rel else None,
                    "mask_status": mask_status,
                    "dynamic_classes": DYNAMIC_CLASSES,
                    "qa_status": qa_status,
                    "timestamp_s": None,
                }
            )

    return {
        "schema_version": "0.1",
        "case_id": args.case_id,
        "source": {
            "kind": kind,
            "path": str(source),
            "camera": args.camera,
            "lidar_used": False,
        },
        "frame_range": [args.frame_start, args.frame_end],
        "notes": [
            "RGB-only demo manifest.",
            "Placeholder masks are smoke-test artifacts and are not accident evidence.",
            "Replace placeholder masks with Grounded SAM 2 or SAM 2 outputs before reconstruction claims.",
        ],
        "frames": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a camera-only RGB demo frame set.")
    parser.add_argument("--source", required=True, help="viewer_frames root or DeepAccident dataset root.")
    parser.add_argument("--source-kind", choices=["auto", "viewer_frames", "deepaccident_rgb"], default="auto")
    parser.add_argument("--out", required=True, help="Output case directory.")
    parser.add_argument("--case-id", default="rgb_demo")
    parser.add_argument("--agents", nargs="*", default=DEFAULT_AGENTS)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=12)
    parser.add_argument("--camera", default="Camera_Front")
    parser.add_argument("--category", default="type1_subtype1_accident")
    parser.add_argument("--scenario", default="Town03_type001_subtype0001_scenario00024")
    parser.add_argument("--copy", action="store_true", help="Copy frames instead of creating relative symlinks.")
    parser.add_argument("--make-empty-masks", action="store_true", help="Create black placeholder masks for smoke testing.")
    args = parser.parse_args()

    if args.frame_end < args.frame_start:
        raise SystemExit("--frame-end must be >= --frame-start")

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    manifest_path = out / "mask_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "frames": len(manifest["frames"]),
        "source_kind": manifest["source"]["kind"],
        "lidar_used": manifest["source"]["lidar_used"],
    }, indent=2))


if __name__ == "__main__":
    main()
