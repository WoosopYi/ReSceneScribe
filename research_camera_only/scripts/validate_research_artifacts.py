#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "lidar_claim_audit.md",
    "technology_decision_record.md",
    "pipeline_design.md",
    "experiment_protocol.md",
    "implementation_skeleton.md",
    "reports/validation_checklist.md",
    "reports/completion_audit.md",
    "reports/limitations.md",
    "reports/evidence_report.md",
    "schemas/mask_manifest.schema.json",
    "schemas/cameras.schema.json",
    "schemas/scale_alignment.schema.json",
    "schemas/vehicle_tracks.schema.json",
    "schemas/collision_diagnostics.schema.json",
    "schemas/evidence_bundle.schema.json",
    "scripts/prepare_rgb_demo.py",
    "scripts/compute_obb_diagnostics.py",
]


def validate_manifest(path: Path) -> list[str]:
    errors: list[str] = []
    data = json.loads(path.read_text())
    for key in ["schema_version", "case_id", "source", "frames"]:
        if key not in data:
            errors.append(f"manifest missing key: {key}")
    source = data.get("source", {})
    if source.get("lidar_used") is not False:
        errors.append("manifest source.lidar_used must be false")
    frames = data.get("frames", [])
    if not isinstance(frames, list) or not frames:
        errors.append("manifest frames must be a non-empty list")
        return errors
    base = path.parent
    for idx, frame in enumerate(frames):
        for key in ["agent", "frame_index", "rgb_path", "mask_status", "dynamic_classes", "qa_status"]:
            if key not in frame:
                errors.append(f"frame {idx} missing key: {key}")
        rgb = frame.get("rgb_path")
        if rgb and not (base / rgb).exists():
            errors.append(f"frame {idx} rgb_path does not exist: {rgb}")
        mask = frame.get("mask_path")
        if mask and not (base / mask).exists():
            errors.append(f"frame {idx} mask_path does not exist: {mask}")
        if frame.get("mask_status") == "placeholder_empty" and frame.get("qa_status") != "not_evidence":
            errors.append(f"frame {idx} placeholder mask must be qa_status=not_evidence")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate camera-only research artifacts.")
    parser.add_argument("--root", default="research_camera_only")
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    root = Path(args.root)
    errors: list[str] = []
    for rel in REQUIRED_FILES:
        path = root / rel
        if not path.exists():
            errors.append(f"missing required file: {rel}")
        elif path.is_file() and path.stat().st_size == 0:
            errors.append(f"empty required file: {rel}")

    if args.manifest:
        errors.extend(validate_manifest(Path(args.manifest)))

    result = {
        "root": str(root),
        "required_files": len(REQUIRED_FILES),
        "manifest": args.manifest or None,
        "status": "pass" if not errors else "fail",
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
