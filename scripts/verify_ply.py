#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


EXPECTED_FIELDS = [
    'x', 'y', 'z',
    'nx', 'ny', 'nz',
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    'opacity',
    'scale_0', 'scale_1', 'scale_2',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
    'red', 'green', 'blue',
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, with_hash: bool) -> dict:
    ply = PlyData.read(str(path))
    vertex = ply['vertex'].data
    names = vertex.dtype.names or ()
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float64)
    rgb = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1).astype(np.uint8)
    float_fields = [name for name in names if vertex[name].dtype.kind == 'f']
    finite_float_fields = all(np.isfinite(vertex[name]).all() for name in float_fields)

    result = {
        'path': str(path),
        'size_bytes': path.stat().st_size,
        'size_mib': round(path.stat().st_size / 1024 / 1024, 2),
        'vertices': int(len(vertex)),
        'properties': len(names),
        'has_expected_fields': all(name in names for name in EXPECTED_FIELDS),
        'has_3dgs_fields': all(name in names for name in ['f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']),
        'finite_xyz': bool(np.isfinite(xyz).all()),
        'finite_float_fields': bool(finite_float_fields),
        'bbox_min': xyz.min(axis=0).tolist(),
        'bbox_max': xyz.max(axis=0).tolist(),
        'rgb_mean': rgb.mean(axis=0).tolist(),
        'rgb_std': rgb.std(axis=0).tolist(),
    }
    if with_hash:
        result['sha256'] = sha256(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Verify a ReSceneScribe hybrid Gaussian/RGB PLY.')
    parser.add_argument('ply', type=Path)
    parser.add_argument('--hash', action='store_true', help='Also compute SHA-256. This reads the file twice.')
    args = parser.parse_args()
    print(json.dumps(verify(args.ply, args.hash), indent=2))


if __name__ == '__main__':
    main()
