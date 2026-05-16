#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from plyfile import PlyData

from gsplat.exporter import export_splats
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import remove as remove_gaussians


SH_C0 = 0.28209479177387814


@dataclass
class ViewItem:
    view_id: str
    image_path: Path
    mask_path: Path
    agent: str
    camera: str
    frame: int
    c2w: np.ndarray
    K: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a real image-based 3D Gaussian Splatting scene for DeepAccident Town04.")
    parser.add_argument("--calibrated-output", type=Path, default=Path("outputs/town04_type1_subtype2_multicam_export"))
    parser.add_argument("--init-ply", type=Path, default=Path("outputs/town04_type1_subtype2_reconstruction/reconstruction/points_world.ply"))
    parser.add_argument("--out", type=Path, default=Path("outputs/town04_type1_subtype2_gsplat"))
    parser.add_argument("--train-width", type=int, default=640)
    parser.add_argument("--max-views", type=int, default=312)
    parser.add_argument("--num-splats", type=int, default=120000)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--init-scale-m", type=float, default=0.30)
    parser.add_argument("--init-opacity", type=float, default=0.12)
    parser.add_argument("--scale-reg", type=float, default=0.002)
    parser.add_argument("--opacity-reg", type=float, default=0.0002)
    parser.add_argument("--lr-means", type=float, default=0.0012)
    parser.add_argument("--lr-scales", type=float, default=0.004)
    parser.add_argument("--lr-opacity", type=float, default=0.018)
    parser.add_argument("--lr-colors", type=float, default=0.012)
    parser.add_argument("--lr-quats", type=float, default=0.0008)
    parser.add_argument("--near", type=float, default=0.02)
    parser.add_argument("--far", type=float, default=8.0)
    parser.add_argument("--orbit-frames", type=int, default=48)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--no-mask-loss", action="store_true")
    parser.add_argument("--no-densify", action="store_true")
    parser.add_argument("--refine-start-iter", type=int, default=250)
    parser.add_argument("--refine-stop-iter", type=int, default=2600)
    parser.add_argument("--refine-every", type=int, default=100)
    parser.add_argument("--reset-every", type=int, default=1200)
    parser.add_argument("--grow-grad2d", type=float, default=0.0008)
    parser.add_argument("--grow-scale3d", type=float, default=0.010)
    parser.add_argument("--grow-scale2d", type=float, default=0.050)
    parser.add_argument("--prune-opa", type=float, default=0.004)
    parser.add_argument("--prune-scale3d", type=float, default=0.120)
    parser.add_argument("--prune-scale2d", type=float, default=0.150)
    parser.add_argument("--max-gaussians", type=int, default=900000)
    return parser.parse_args()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def logit_np(x: np.ndarray | float) -> np.ndarray:
    x = np.clip(x, 1e-4, 1.0 - 1e-4)
    return np.log(x / (1.0 - x))


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    return torch.clamp(sh0 * SH_C0 + 0.5, 0.0, 1.0)


def standard_intrinsics(raw_k: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
    fx = abs(float(raw_k[0, 1])) if abs(float(raw_k[0, 1])) > 1.0 else abs(float(raw_k[1, 2]))
    fy = abs(float(raw_k[1, 2])) if abs(float(raw_k[1, 2])) > 1.0 else fx
    cx = float(raw_k[0, 0]) if abs(float(raw_k[0, 0])) > 1.0 else width / 2.0
    cy = float(raw_k[1, 0]) if abs(float(raw_k[1, 0])) > 1.0 else height / 2.0
    K = np.array([[fx * scale, 0.0, cx * scale], [0.0, fy * scale, cy * scale], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def load_views(calibrated_output: Path, train_width: int, max_views: int) -> tuple[list[ViewItem], int, int]:
    cameras = json.loads((calibrated_output / "reconstruction" / "cameras.json").read_text(encoding="utf-8"))
    selected = cameras["views"][:max_views] if max_views > 0 else cameras["views"]
    views: list[ViewItem] = []
    train_height = None
    for raw in selected:
        image_path = calibrated_output / raw["rgb_path"]
        if not image_path.exists():
            continue
        with Image.open(image_path) as img:
            width, height = img.size
        scale = train_width / float(width)
        resized_height = int(round(height * scale))
        train_height = resized_height if train_height is None else train_height
        views.append(
            ViewItem(
                view_id=raw["view_id"],
                image_path=image_path,
                mask_path=calibrated_output / raw["mask_path"],
                agent=raw["agent"],
                camera=raw["camera"],
                frame=int(raw["frame"]),
                c2w=np.asarray(raw["camera_to_world_cv"], dtype=np.float32),
                K=standard_intrinsics(np.asarray(raw["intrinsic_raw"], dtype=np.float32), width, height, scale),
            )
        )
    if not views or train_height is None:
        raise RuntimeError(f"No RGB views found under {calibrated_output}")
    return views, train_width, train_height


def load_image_tensor(path: Path, width: int, height: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def load_mask_tensor(path: Path, width: int, height: int, disabled: bool) -> torch.Tensor:
    if disabled or not path.exists():
        return torch.ones((height, width, 1), dtype=torch.float32)
    mask = Image.open(path).convert("L").resize((width, height), Image.Resampling.NEAREST)
    arr = np.asarray(mask, dtype=np.uint8)
    valid = (arr < 8).astype(np.float32)
    if valid.mean() < 0.05:
        valid = np.ones_like(valid, dtype=np.float32)
    return torch.from_numpy(valid[..., None])


def load_training_tensors(views: list[ViewItem], width: int, height: int, no_mask_loss: bool, device: torch.device) -> dict:
    images = torch.stack([load_image_tensor(v.image_path, width, height) for v in views]).to(device)
    masks = torch.stack([load_mask_tensor(v.mask_path, width, height, no_mask_loss) for v in views]).to(device)
    c2ws = torch.from_numpy(np.stack([v.c2w for v in views]).astype(np.float32)).to(device)
    Ks = torch.from_numpy(np.stack([v.K for v in views]).astype(np.float32)).to(device)
    return {"images": images, "masks": masks, "c2ws_world": c2ws, "Ks": Ks}


def read_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
    colors = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype(np.float32) / 255.0
    return points, colors


def build_scene_normalization(points: np.ndarray, c2ws: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    camera_centers = c2ws[:, :3, 3]
    origin = np.median(camera_centers, axis=0).astype(np.float32)
    point_radii = np.linalg.norm(points - origin[None, :], axis=1)
    cam_radii = np.linalg.norm(camera_centers - origin[None, :], axis=1)
    scale_m = float(max(8.0, np.percentile(np.concatenate([point_radii, cam_radii]), 92)))
    keep = point_radii < scale_m * 1.65
    return origin, scale_m, keep


def initialize_gaussians(args: argparse.Namespace, views: list[ViewItem], device: torch.device) -> tuple[dict[str, torch.nn.Parameter], dict]:
    points, colors = read_ply_points(args.init_ply)
    c2ws = np.stack([v.c2w for v in views]).astype(np.float32)
    origin, scale_m, keep = build_scene_normalization(points, c2ws)
    points = points[keep]
    colors = colors[keep]
    if len(points) == 0:
        raise RuntimeError("Image-based initialization produced zero points after robust filtering.")
    rng = np.random.default_rng(args.seed)
    if len(points) > args.num_splats:
        idx = rng.choice(len(points), size=args.num_splats, replace=False)
        points = points[idx]
        colors = colors[idx]
    means_norm = (points - origin[None, :]) / scale_m
    init_scale = max(args.init_scale_m / scale_m, 1e-4)
    n = len(means_norm)
    params = {
        "means": torch.nn.Parameter(torch.from_numpy(means_norm).to(device)),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(init_scale), dtype=torch.float32, device=device)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.full((n,), float(logit_np(args.init_opacity)), dtype=torch.float32, device=device)),
        "colors": torch.nn.Parameter(torch.from_numpy(logit_np(colors)).to(device)),
    }
    report = {
        "init_source": str(args.init_ply),
        "init_source_type": "image_based_vggt_dense_points_for_gaussian_initialization_only",
        "initial_splats": int(n),
        "scene_origin_world": origin.tolist(),
        "scene_scale_m": scale_m,
        "robust_points_kept": int(len(points)),
        "init_scale_m": args.init_scale_m,
        "init_opacity": args.init_opacity,
    }
    return params, report


def normalized_viewmats(c2ws_world: torch.Tensor, origin_world: torch.Tensor, scale_m: float) -> torch.Tensor:
    c2ws = c2ws_world.clone()
    c2ws[:, :3, 3] = (c2ws[:, :3, 3] - origin_world[None, :]) / scale_m
    return torch.linalg.inv(c2ws)


def render_batch(
    params: dict[str, torch.nn.Parameter],
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    near: float,
    far: float,
    return_info: bool = False,
    absgrad: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    means = params["means"]
    quats = params["quats"] / torch.clamp(torch.linalg.norm(params["quats"], dim=-1, keepdim=True), min=1e-8)
    scales = torch.exp(torch.clamp(params["scales"], min=-10.0, max=-1.0))
    opacities = torch.sigmoid(params["opacities"])
    colors = torch.sigmoid(params["colors"])
    renders, _alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        radius_clip=0.0,
        render_mode="RGB",
        packed=True,
        backgrounds=None,
        absgrad=absgrad,
    )
    if return_info:
        return renders, meta
    return renders


def masked_psnr(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    mse = (((render - target) ** 2) * mask).sum() / torch.clamp(mask.sum() * 3.0, min=1.0)
    return float(-10.0 * torch.log10(torch.clamp(mse, min=1e-8)).detach().cpu())


def save_image(path: Path, tensor: torch.Tensor) -> None:
    arr = torch.clamp(tensor.detach().cpu(), 0.0, 1.0).numpy()
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)


def save_compare(path: Path, target: torch.Tensor, render: torch.Tensor, mask: torch.Tensor, label: str) -> None:
    target_img = Image.fromarray((torch.clamp(target.detach().cpu(), 0, 1).numpy() * 255).astype(np.uint8))
    render_img = Image.fromarray((torch.clamp(render.detach().cpu(), 0, 1).numpy() * 255).astype(np.uint8))
    mask_rgb = Image.fromarray((mask.detach().cpu().repeat(1, 1, 3).numpy() * 255).astype(np.uint8))
    canvas = Image.new("RGB", (target_img.width * 3, target_img.height + 28), (18, 18, 18))
    canvas.paste(target_img, (0, 28))
    canvas.paste(render_img, (target_img.width, 28))
    canvas.paste(mask_rgb, (target_img.width * 2, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), f"target | gaussian render | loss mask  -  {label}", fill=(238, 238, 238))
    canvas.save(path)


def export_gaussian_assets(out: Path, params: dict[str, torch.nn.Parameter], origin_world: np.ndarray, scale_m: float) -> None:
    with torch.no_grad():
        quats = params["quats"] / torch.clamp(torch.linalg.norm(params["quats"], dim=-1, keepdim=True), min=1e-8)
        rgb = torch.sigmoid(params["colors"])
        sh0 = rgb_to_sh(rgb).unsqueeze(1)
        shn = torch.empty((rgb.shape[0], 0, 3), dtype=rgb.dtype, device=rgb.device)
        means_norm = params["means"]
        log_scales_norm = params["scales"]
        opacities = params["opacities"]
        export_splats(means_norm, log_scales_norm, quats, opacities, sh0, shn, format="ply", save_to=str(out / "gaussians_normalized.ply"))
        export_splats(means_norm, log_scales_norm, quats, opacities, sh0, shn, format="splat", save_to=str(out / "gaussians_normalized.splat"))
        origin = torch.tensor(origin_world, dtype=means_norm.dtype, device=means_norm.device)
        means_world = means_norm * scale_m + origin[None, :]
        log_scales_world = log_scales_norm + math.log(scale_m)
        export_splats(means_world, log_scales_world, quats, opacities, sh0, shn, format="ply", save_to=str(out / "gaussians_world.ply"))


def look_at_c2w(camera_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - camera_pos
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-8)
    down = np.cross(forward, right)
    down = down / max(np.linalg.norm(down), 1e-8)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_pos
    return c2w


def render_outputs(args: argparse.Namespace, out: Path, params: dict[str, torch.nn.Parameter], tensors: dict, views: list[ViewItem], init_report: dict) -> dict:
    render_dir = out / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    device = params["means"].device
    origin_world = torch.tensor(init_report["scene_origin_world"], dtype=torch.float32, device=device)
    scale_m = float(init_report["scene_scale_m"])
    viewmats_all = normalized_viewmats(tensors["c2ws_world"], origin_world, scale_m)
    eval_indices = sorted(set([0, len(views) // 3, len(views) // 2, len(views) - 1]))
    closest_candidates = [i for i, v in enumerate(views) if v.frame == 49 and v.camera == "Camera_Front"]
    eval_indices.extend(closest_candidates[:4])
    eval_indices = sorted(set(i for i in eval_indices if 0 <= i < len(views)))
    view_reports = []
    with torch.no_grad():
        for idx in eval_indices:
            render = render_batch(params, viewmats_all[idx : idx + 1], tensors["Ks"][idx : idx + 1], args.train_width, tensors["images"].shape[1], args.near, args.far)[0]
            target = tensors["images"][idx]
            mask = tensors["masks"][idx]
            psnr = masked_psnr(render, target, mask)
            label = f"{views[idx].view_id} psnr={psnr:.2f}dB"
            save_compare(render_dir / f"compare_{idx:04d}.png", target, render, mask, label)
            save_image(render_dir / f"render_{idx:04d}.png", render)
            view_reports.append({"index": int(idx), "view_id": views[idx].view_id, "psnr": psnr})

        camera_centers = tensors["c2ws_world"][:, :3, 3].detach().cpu().numpy()
        target_world = np.median(camera_centers, axis=0)
        radius = max(np.percentile(np.linalg.norm(camera_centers - target_world[None, :], axis=1), 90), scale_m * 0.65)
        height = max(scale_m * 0.23, 8.0)
        render_height = int(round(args.render_width * tensors["images"].shape[1] / tensors["images"].shape[2]))
        fov_fx = float(torch.median(tensors["Ks"][:, 0, 0]).detach().cpu())
        fov_scale = args.render_width / float(args.train_width)
        K_orbit = torch.tensor(
            [[[fov_fx * fov_scale, 0.0, args.render_width / 2.0], [0.0, fov_fx * fov_scale, render_height / 2.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
            device=device,
        )
        orbit_paths = []
        for frame in range(args.orbit_frames):
            theta = 2.0 * math.pi * frame / max(args.orbit_frames, 1)
            camera_pos = target_world + np.array([math.cos(theta) * radius, math.sin(theta) * radius, height], dtype=np.float32)
            c2w_world = look_at_c2w(camera_pos, target_world, np.array([0.0, 0.0, 1.0], dtype=np.float32))
            c2w = torch.from_numpy(c2w_world[None]).to(device)
            viewmat = normalized_viewmats(c2w, origin_world, scale_m)
            render = render_batch(params, viewmat, K_orbit, args.render_width, render_height, args.near, args.far)[0]
            path = render_dir / f"orbit_{frame:03d}.png"
            save_image(path, render)
            orbit_paths.append(path.name)

    write_html_viewer(out, orbit_paths, view_reports)
    return {"eval_views": view_reports, "orbit_frames": len(orbit_paths), "render_dir": "renders"}


def write_html_viewer(out: Path, orbit_paths: list[str], view_reports: list[dict]) -> None:
    viewer = out / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    compares = [f"../renders/compare_{item['index']:04d}.png" for item in view_reports]
    orbit_json = json.dumps([f"../renders/{name}" for name in orbit_paths])
    compare_json = json.dumps(compares)
    (viewer / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Town04 Gaussian Splat Reconstruction</title>
  <style>
    html, body {{ margin:0; min-height:100%; background:#08090b; color:#e8eaed; font:14px system-ui, sans-serif; }}
    header {{ position:sticky; top:0; z-index:2; display:flex; flex-wrap:wrap; gap:10px; align-items:center; padding:10px 14px; background:#11151a; border-bottom:1px solid #2b333d; }}
    button {{ background:#1f2937; color:#e8eaed; border:1px solid #3b4654; padding:7px 10px; cursor:pointer; }}
    input {{ width:min(460px, 60vw); }}
    main {{ display:grid; grid-template-columns:minmax(0, 1fr); gap:12px; padding:12px; }}
    img {{ max-width:100%; height:auto; background:#000; display:block; }}
    #stage {{ width:100%; max-height:calc(100vh - 92px); object-fit:contain; }}
    .meta {{ color:#aeb8c4; }}
  </style>
</head>
<body>
  <header>
    <strong>Image-trained Gaussian Splatting reconstruction</strong>
    <button id="orbit">orbit</button>
    <button id="compare">camera compares</button>
    <button id="play">play</button>
    <input id="slider" type="range" min="0" value="0" step="1" />
    <span id="label" class="meta"></span>
  </header>
  <main><img id="stage" alt="Gaussian splat render" /></main>
  <script>
    const orbit = {orbit_json};
    const compares = {compare_json};
    let mode = 'orbit';
    let frames = orbit;
    let idx = 0;
    let timer = null;
    const img = document.getElementById('stage');
    const slider = document.getElementById('slider');
    const label = document.getElementById('label');
    function show(i) {{
      idx = Math.max(0, Math.min(frames.length - 1, i));
      slider.max = Math.max(0, frames.length - 1);
      slider.value = idx;
      img.src = frames[idx];
      label.textContent = `${{mode}} ${{idx + 1}} / ${{frames.length}}`;
    }}
    function setMode(next) {{
      mode = next;
      frames = mode === 'orbit' ? orbit : compares;
      show(0);
    }}
    document.getElementById('orbit').onclick = () => setMode('orbit');
    document.getElementById('compare').onclick = () => setMode('compare');
    document.getElementById('play').onclick = () => {{
      if (timer) {{ clearInterval(timer); timer = null; return; }}
      timer = setInterval(() => show((idx + 1) % frames.length), 130);
    }};
    slider.oninput = () => show(Number(slider.value));
    show(0);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def make_optimizers(args: argparse.Namespace, params: dict[str, torch.nn.Parameter]) -> dict[str, torch.optim.Optimizer]:
    lrs = {
        "means": args.lr_means,
        "scales": args.lr_scales,
        "opacities": args.lr_opacity,
        "colors": args.lr_colors,
        "quats": args.lr_quats,
    }
    return {name: torch.optim.Adam([params[name]], lr=lr, eps=1e-15) for name, lr in lrs.items()}


def make_strategy(args: argparse.Namespace, params: dict[str, torch.nn.Parameter], optimizers: dict[str, torch.optim.Optimizer]) -> tuple[DefaultStrategy | None, dict]:
    if args.no_densify:
        return None, {}
    strategy = DefaultStrategy(
        prune_opa=args.prune_opa,
        grow_grad2d=args.grow_grad2d,
        grow_scale3d=args.grow_scale3d,
        grow_scale2d=args.grow_scale2d,
        prune_scale3d=args.prune_scale3d,
        prune_scale2d=args.prune_scale2d,
        refine_start_iter=args.refine_start_iter,
        refine_stop_iter=args.refine_stop_iter,
        reset_every=args.reset_every,
        refine_every=args.refine_every,
        absgrad=True,
        verbose=True,
    )
    strategy.check_sanity(params, optimizers)
    return strategy, strategy.initialize_state(scene_scale=1.0)


@torch.no_grad()
def cap_gaussians(
    params: dict[str, torch.nn.Parameter],
    optimizers: dict[str, torch.optim.Optimizer],
    strategy_state: dict,
    max_gaussians: int,
) -> int:
    if max_gaussians <= 0:
        return 0
    n = int(params["means"].shape[0])
    if n <= max_gaussians:
        return 0
    prune_count = n - max_gaussians
    opacities = torch.sigmoid(params["opacities"].flatten())
    prune_idx = torch.topk(opacities, k=prune_count, largest=False).indices
    prune_mask = torch.zeros(n, dtype=torch.bool, device=opacities.device)
    prune_mask[prune_idx] = True
    remove_gaussians(params=params, optimizers=optimizers, state=strategy_state, mask=prune_mask)
    return int(prune_count)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat training.")
    device = torch.device("cuda")
    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    for rel in ["renders", "viewer", "checkpoints", "reports"]:
        (out / rel).mkdir(parents=True, exist_ok=True)

    views, width, height = load_views(args.calibrated_output.resolve(), args.train_width, args.max_views)
    tensors = load_training_tensors(views, width, height, args.no_mask_loss, device)
    params, init_report = initialize_gaussians(args, views, device)
    origin_world = torch.tensor(init_report["scene_origin_world"], dtype=torch.float32, device=device)
    viewmats = normalized_viewmats(tensors["c2ws_world"], origin_world, init_report["scene_scale_m"])

    optimizers = make_optimizers(args, params)
    strategy, strategy_state = make_strategy(args, params, optimizers)
    logs = []
    num_views = len(views)
    for step in range(1, args.steps + 1):
        idx = random.randrange(num_views)
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        render, info = render_batch(
            params,
            viewmats[idx : idx + 1],
            tensors["Ks"][idx : idx + 1],
            width,
            height,
            args.near,
            args.far,
            return_info=True,
            absgrad=strategy.absgrad if strategy is not None else False,
        )
        target = tensors["images"][idx : idx + 1]
        mask = tensors["masks"][idx : idx + 1]
        l1 = (torch.abs(render - target) * mask).sum() / torch.clamp(mask.sum() * 3.0, min=1.0)
        mse = (((render - target) ** 2) * mask).sum() / torch.clamp(mask.sum() * 3.0, min=1.0)
        scale_reg = torch.exp(params["scales"]).mean() * args.scale_reg
        opacity_reg = torch.sigmoid(params["opacities"]).mean() * args.opacity_reg
        loss = 0.82 * l1 + 0.18 * mse + scale_reg + opacity_reg
        if strategy is not None:
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        loss.backward()
        if strategy is not None:
            strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)
            cap_gaussians(params, optimizers, strategy_state, args.max_gaussians)
        for opt in optimizers.values():
            opt.step()
        with torch.no_grad():
            params["quats"].data = params["quats"].data / torch.clamp(torch.linalg.norm(params["quats"].data, dim=-1, keepdim=True), min=1e-8)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            psnr = masked_psnr(render[0], target[0], mask[0])
            log = {
                "step": step,
                "view_index": int(idx),
                "view_id": views[idx].view_id,
                "loss": float(loss.detach().cpu()),
                "l1": float(l1.detach().cpu()),
                "mse": float(mse.detach().cpu()),
                "psnr": psnr,
                "gaussians": int(params["means"].shape[0]),
                "mean_opacity": float(torch.sigmoid(params["opacities"]).mean().detach().cpu()),
                "mean_scale_m": float((torch.exp(params["scales"]).mean() * init_report["scene_scale_m"]).detach().cpu()),
            }
            logs.append(log)
            print(json.dumps(log), flush=True)

    checkpoint = {
        "params": {k: v.detach().cpu() for k, v in params.items()},
        "init_report": init_report,
        "width": width,
        "height": height,
        "calibrated_output": str(args.calibrated_output.resolve()),
        "views": [v.__dict__ | {"image_path": str(v.image_path), "mask_path": str(v.mask_path), "c2w": v.c2w.tolist(), "K": v.K.tolist()} for v in views],
    }
    torch.save(checkpoint, out / "checkpoints" / "checkpoint_last.pt")
    export_gaussian_assets(out, params, np.asarray(init_report["scene_origin_world"], dtype=np.float32), float(init_report["scene_scale_m"]))
    render_report = render_outputs(args, out, params, tensors, views, init_report)

    report = {
        "schema_version": "0.1",
        "status": "ok",
        "method": "image_based_3d_gaussian_splatting_gsplat",
        "lidar_used": False,
        "proxy_mesh_used": False,
        "dataset": {
            "calibrated_output": str(args.calibrated_output.resolve()),
            "rgb_views": len(views),
            "train_resolution": [width, height],
            "dynamic_masks_used": not args.no_mask_loss,
        },
        "gaussians": {
            "count": int(params["means"].shape[0]),
            "exports": {
                "world_ply": "gaussians_world.ply",
                "normalized_ply": "gaussians_normalized.ply",
                "normalized_splat": "gaussians_normalized.splat",
                "checkpoint": "checkpoints/checkpoint_last.pt",
            },
        },
        "initialization": init_report,
        "training": {
            "steps": args.steps,
            "densification": {
                "enabled": strategy is not None,
                "refine_start_iter": args.refine_start_iter,
                "refine_stop_iter": args.refine_stop_iter,
                "refine_every": args.refine_every,
                "grow_grad2d": args.grow_grad2d,
                "max_gaussians": args.max_gaussians,
            },
            "logs": logs,
            "final": logs[-1] if logs else None,
        },
        "renders": render_report,
        "viewer": "viewer/index.html",
    }
    write_json(out / "manifest.json", report)
    (out / "reports" / "no_lidar_audit.md").write_text(
        f"""# No-LiDAR Gaussian Splat Audit

Status: `ok`

- Final representation: 3D Gaussian Splatting parameters, not point-cloud viewer and not proxy mesh.
- RGB training views: `{len(views)}`
- Calibration camera poses used: `true`
- Dynamic object masks used in loss: `{str(not args.no_mask_loss).lower()}`
- LiDAR geometry used: `false`
- `lidar01` files read: `false`
- Image-based VGGT points used only as Gaussian initialization: `true`
- Exported Gaussian count: `{int(params['means'].shape[0])}`
- Densification/split/prune strategy: `{str(strategy is not None).lower()}`

The train/render loss is computed by differentiably rasterizing Gaussians back
into the RGB camera frames with `gsplat`. The output viewer displays rendered
Gaussian images, not point sprites or hand-made mesh proxies.
""",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
