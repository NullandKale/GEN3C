# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
import cv2
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import time
from PIL import Image
from contextlib import contextmanager
from typing import Tuple

from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import save_video

torch.enable_grad(False)


@contextmanager
def _timed(section: str):
    t0 = time.perf_counter()
    log.info(f"[TIMER] {section} | start")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log.info(f"[TIMER] {section} | end: {dt:.3f}s")


def _ensure_dir(d: str | None):
    if d:
        os.makedirs(d, exist_ok=True)


def _parse_quilt_meta_from_name(path: str) -> tuple[int | None, int | None, float | None]:
    """
    Extract (cols, rows, aspect) from names like 'foo_qs8x6a0.75.png'.
    Returns (None, None, None) if not found.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"_qs(?P<c>\d+)x(?P<r>\d+)(?:a(?P<a>[0-9]*\.?[0-9]+))?", stem, flags=re.IGNORECASE)
    if not m:
        return None, None, None
    c = int(m.group("c"))
    r = int(m.group("r"))
    a = m.group("a")
    ar = float(a) if a is not None else None
    return c, r, ar


def _extract_quilt_frames(
    quilt_path: str,
    grid_cols: int,
    grid_rows: int,
    total_frames: int | None,
    order: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
    gap_x: int,
    gap_y: int
) -> list[np.ndarray]:
    """
    Extract frames from a single quilt image into a list of RGB frames.
    order:
      - 'quilt'       : row-major with bottom-left origin (Looking Glass convention)
      - 'row-major'   : row-major with top-left origin
      - 'col-major'   : col-major with top-left origin
    """
    img_bgr = cv2.imread(quilt_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Quilt image not found: {quilt_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    work_w = W - left - right
    work_h = H - top - bottom
    if work_w <= 0 or work_h <= 0:
        raise ValueError(f"Invalid crop margins; got work area {work_w}x{work_h}.")
    if grid_cols <= 0 or grid_rows <= 0:
        raise ValueError("grid_cols and grid_rows must be positive.")
    if grid_cols == 1:
        tile_w = work_w
    else:
        tile_w = (work_w - (grid_cols - 1) * gap_x) // grid_cols
    if grid_rows == 1:
        tile_h = work_h
    else:
        tile_h = (work_h - (grid_rows - 1) * gap_y) // grid_rows
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError(f"Computed non-positive tile size {tile_w}x{tile_h}. Check gaps/margins.")

    idxs = []
    if order == "quilt":
        for r in range(grid_rows - 1, -1, -1):
            for c in range(grid_cols):
                idxs.append((r, c))
    elif order == "row-major":
        for r in range(grid_rows):
            for c in range(grid_cols):
                idxs.append((r, c))
    elif order == "col-major":
        for c in range(grid_cols):
            for r in range(grid_rows):
                idxs.append((r, c))
    else:
        raise ValueError("order must be 'quilt', 'row-major', or 'col-major'.")

    max_frames = grid_cols * grid_rows
    T = total_frames if total_frames is not None else max_frames
    T = max(0, min(T, max_frames))

    frames = []
    for k, (r, c) in enumerate(idxs):
        if k >= T:
            break
        x0 = left + c * (tile_w + gap_x)
        y0 = top + r * (tile_h + gap_y)
        crop = img_rgb[y0:y0 + tile_h, x0:x0 + tile_w, :]
        if crop.shape[0] != tile_h or crop.shape[1] != tile_w:
            continue
        frames.append(crop.copy())
    return frames


# ------------------ pose/intrinsics/warping utilities ------------------ #

def _load_meta(sidecar_meta_path: str) -> dict:
    if not os.path.exists(sidecar_meta_path):
        raise FileNotFoundError(
            f"Missing meta sidecar: {sidecar_meta_path}\n"
            f"Expected alongside the quilt. Generate it in the quilt renderer."
        )
    with open(sidecar_meta_path, "r") as f:
        return json.load(f)


def _compute_intrinsics(W: int, H: int, fov_deg: float) -> np.ndarray:
    f = 0.5 * W / max(math.tan(math.radians(fov_deg) * 0.5), 1e-8)
    fx = fy = f
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)


def _tile_order_indices(cols: int, rows: int, order: str) -> list[tuple[int, int]]:
    idxs = []
    if order == "quilt":  # bottom-left origin, row-major
        for r in range(rows - 1, -1, -1):
            for c in range(cols):
                idxs.append((r, c))
    elif order == "row-major":
        for r in range(rows):
            for c in range(cols):
                idxs.append((r, c))
    elif order == "col-major":
        for c in range(cols):
            for r in range(rows):
                idxs.append((r, c))
    else:
        raise ValueError("order must be 'quilt', 'row-major', or 'col-major'.")
    return idxs


def _center_tile_index(cols: int, rows: int, order: str) -> int:
    rc_order = _tile_order_indices(cols, rows, order)
    r_cen, c_cen = rows // 2, cols // 2
    try:
        return rc_order.index((r_cen, c_cen))
    except ValueError:
        return len(rc_order) // 2  # fallback


def _lkg_w2c_list(cols: int, rows: int, fov_deg: float, viewcone_deg: float, camera_size: float, invert_quilt: bool, order: str) -> np.ndarray:
    # Distance and lateral offset
    cam_dist = camera_size / max(math.tan(math.radians(fov_deg)), 1e-8)
    cam_off = cam_dist * math.tan(math.radians(viewcone_deg))
    rc_order = _tile_order_indices(cols, rows, order)
    V = len(rc_order)

    w2c_list = []
    for k, _ in enumerate(rc_order):
        vnorm = (k + 0.5) / V
        vfc = vnorm - 0.5
        offset = -(vfc) * cam_off

        # Basis
        s = np.array([1, 0, 0], dtype=np.float32)
        u = np.array([0, -1 if invert_quilt else 1, 0], dtype=np.float32)
        f = np.array([0, 0, 1], dtype=np.float32)

        R = np.eye(4, dtype=np.float32)
        R[0, 0], R[1, 0], R[2, 0] = s
        R[0, 1], R[1, 1], R[2, 1] = u
        R[0, 2], R[1, 2], R[2, 2] = -f

        T = np.eye(4, dtype=np.float32)
        T[0, 3] = offset
        T[2, 3] = -cam_dist

        w2c = R @ T
        w2c_list.append(w2c)
    return np.stack(w2c_list, axis=0)  # (T,4,4)


def _depth01_to_meters(depth01: np.ndarray, near_m: float = 0.3, far_m: float = 3.0) -> np.ndarray:
    # WHITE=NEAR => 1.0 near, 0.0 far
    return near_m + (1.0 - depth01) * (far_m - near_m)


def _warp_center_to_view(center_rgb: np.ndarray,
                         center_depth_m: np.ndarray,
                         K_np: np.ndarray,
                         w2c_center_np: np.ndarray,
                         w2c_view_np: np.ndarray,
                         device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    center_rgb: HxWx3 uint8
    center_depth_m: HxW float32 (meters)
    K_np: 3x3
    w2c_center_np, w2c_view_np: 4x4
    Returns:
        warped_chw in [0,1] (C,H,W) float32
        mask_1hw in {0,1}  (1,H,W) float32
    """
    H, W, _ = center_rgb.shape
    K = torch.from_numpy(K_np).to(device=device, dtype=torch.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    xs = xs.float(); ys = ys.float()

    z = torch.from_numpy(center_depth_m).to(device=device, dtype=torch.float32)  # HxW
    X = (xs - cx) / fx * z
    Y = (ys - cy) / fy * z
    Z = z
    ones = torch.ones_like(Z)
    pc0 = torch.stack([X, Y, Z, ones], dim=-1).reshape(-1, 4).t()  # 4xN

    c2w0 = torch.from_numpy(np.linalg.inv(w2c_center_np)).to(device=device, dtype=torch.float32)
    w2c  = torch.from_numpy(w2c_view_np).to(device=device, dtype=torch.float32)
    pci  = (w2c @ (c2w0 @ pc0))  # 4xN
    Xi, Yi, Zi = pci[0, :], pci[1, :], pci[2, :]

    ui = fx * (Xi / Zi) + cx
    vi = fy * (Yi / Zi) + cy

    valid = (Zi > 1e-6) & (ui >= 0) & (ui <= W - 1) & (vi >= 0) & (vi <= H - 1)
    ui = ui.reshape(H, W); vi = vi.reshape(H, W); valid = valid.reshape(H, W)

    gx = (ui / max(W - 1, 1)) * 2 - 1
    gy = (vi / max(H - 1, 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)[None]  # 1xHxWx2

    cen = torch.from_numpy(center_rgb).to(device=device, dtype=torch.float32) / 255.0
    cen = cen.permute(2, 0, 1)[None]  # 1x3xHxW

    warped = F.grid_sample(cen, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    mask = F.grid_sample(torch.ones_like(cen[:, :1]), grid, mode="nearest", padding_mode="zeros", align_corners=True)
    # AND with geometric validity
    mask = (mask > 0.5).float() * valid.float().unsqueeze(0).unsqueeze(0)

    return warped[0], mask[0]


# --------------------------------------------------------------------------- #

def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GEN3C from quilt: read quilt + sidecars → build warps/masks from center RGBD → generate."
    )
    p.add_argument("--quilt_path", type=str, required=True, help="Path to the quilt image (_qs{C}x{R}... naming recommended).")
    p.add_argument("--grid_cols", type=int, default=0, help="Quilt cols; if 0, parsed from filename.")
    p.add_argument("--grid_rows", type=int, default=0, help="Quilt rows; if 0, parsed from filename.")
    p.add_argument("--total_frames", type=int, default=None, help="Optional cap on frames; default uses all tiles.")
    p.add_argument("--order", type=str, choices=["quilt", "row-major", "col-major"], default="quilt", help="Tile traversal order.")
    p.add_argument("--left", type=int, default=0, help="Left crop.")
    p.add_argument("--top", type=int, default=0, help="Top crop.")
    p.add_argument("--right", type=int, default=0, help="Right crop.")
    p.add_argument("--bottom", type=int, default=0, help="Bottom crop.")
    p.add_argument("--gap_x", type=int, default=0, help="Horizontal gap between tiles.")
    p.add_argument("--gap_y", type=int, default=0, help="Vertical gap between tiles.")

    # Sidecars (NEW: optional explicit paths; if omitted, auto-derive from quilt basename)
    p.add_argument("--meta_json", type=str, default=None, help="Optional path to quilt meta JSON sidecar (.meta.json).")
    p.add_argument("--center_depth_path", type=str, default=None, help="Optional path to center depth PNG16 sidecar (_center_depth.png).")

    # Output / model
    p.add_argument("--checkpoint_dir", type=str, required=True, help="Path to gen3c checkpoint directory (contains Gen3C-Cosmos-7B).")
    p.add_argument("--video_save_folder", type=str, default="outputs", help="Folder for generated outputs.")
    p.add_argument("--video_save_name", type=str, default="gen3c_from_quilt", help="Base name for output MP4.")
    p.add_argument("--height", type=int, default=704, help="Per-tile H (must match model).")
    p.add_argument("--width", type=int, default=1280, help="Per-tile W (must match model).")
    p.add_argument("--fps", type=int, default=24, help="FPS.")
    p.add_argument("--guidance", type=float, default=1.0, help="CFG guidance.")
    p.add_argument("--num_steps", type=int, default=25, help="Diffusion steps.")
    p.add_argument("--seed", type=int, default=1234, help="Random seed.")
    p.add_argument("--disable_guardrail", action="store_true", help="Disable guardrails.")
    p.add_argument("--disable_prompt_upsampler", action="store_true", help="Disable prompt upsampler.")
    p.add_argument("--prompt_upsampler_dir", type=str, default="Pixtral-12B", help="Prompt upsampler dir (relative to checkpoint_dir).")
    p.add_argument("--negative_prompt", type=str, default=None, help="Optional negative prompt.")
    p.add_argument("--prompt", type=str, default="", help="Text prompt (optional).")
    p.add_argument("--offload_diffusion_transformer", action="store_true", help="Offload diffusion transformer.")
    p.add_argument("--offload_tokenizer", action="store_true", help="Offload tokenizer.")
    p.add_argument("--offload_text_encoder_model", action="store_true", help="Offload text encoder.")
    p.add_argument("--offload_prompt_upsampler", action="store_true", help="Offload prompt upsampler.")
    p.add_argument("--offload_guardrail_models", action="store_true", help="Offload guardrail models.")
    p.add_argument("--disable_prompt_encoder", action="store_true", help="Disable prompt encoder.")
    p.add_argument("--save_conditioning_video", action="store_true", help="If set, write the computed conditioning video (warps).")
    p.add_argument("--conditioning_video_name", type=str, default="conditioning_from_quilt.mp4", help="Filename for conditioning video.")
    p.add_argument("--strict_exact_length", action="store_true", help="Require that quilt frames == model chunk_size.")

    # Depth mapping (center depth 1.0=NEAR)
    p.add_argument("--depth_near_m", type=float, default=0.3, help="Near (meters) to map center depth.")
    p.add_argument("--depth_far_m", type=float, default=3.0, help="Far (meters) to map center depth.")

    # Optional: save a small run manifest JSON
    p.add_argument("--save_run_manifest", action="store_true", help="If set, write a JSON manifest next to the output MP4.")
    p.add_argument("--run_manifest_name", type=str, default=None, help="Optional manifest filename (defaults to <video_save_name>.manifest.json).")
    return p


def _resolve_grid_args(quilt_path: str, cols: int, rows: int) -> tuple[int, int]:
    if cols > 0 and rows > 0:
        return cols, rows
    c2, r2, _ = _parse_quilt_meta_from_name(quilt_path)
    if c2 is None or r2 is None:
        raise ValueError("grid_cols/grid_rows not provided and filename does not encode _qs{C}x{R}.")
    log.info(f"Parsed grid from filename: cols={c2}, rows={r2}")
    return c2, r2


def _derive_sidecar_paths(quilt_path: str) -> Tuple[str, str]:
    base, _ext = os.path.splitext(quilt_path)
    meta_path = base + ".meta.json"
    depth_path = base + "_center_depth.png"
    return meta_path, depth_path


def demo(args: argparse.Namespace) -> None:
    t_total_start = time.perf_counter()
    misc.set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} (CUDA={torch.cuda.is_available()})")
    _ensure_dir(args.video_save_folder)

    # --- read quilt frames ---
    cols, rows = _resolve_grid_args(args.quilt_path, args.grid_cols, args.grid_rows)

    with _timed("Unpack quilt"):
        frames_rgb = _extract_quilt_frames(
            quilt_path=args.quilt_path,
            grid_cols=cols,
            grid_rows=rows,
            total_frames=args.total_frames,
            order=args.order,
            left=args.left,
            top=args.top,
            right=args.right,
            bottom=args.bottom,
            gap_x=args.gap_x,
            gap_y=args.gap_y
        )
        log.info(f"Extracted {len(frames_rgb)} frames from quilt (order={args.order}, cols={cols}, rows={rows}).")

    # Resize tiles to target HxW (model resolution)
    frames_rgb = [cv2.resize(f, (args.width, args.height), interpolation=cv2.INTER_AREA) for f in frames_rgb]

    # --- sidecars (meta + center depth) ---
    # Prefer explicit paths if given; otherwise derive from quilt basename.
    meta_path_auto, depth_path_auto = _derive_sidecar_paths(args.quilt_path)
    meta_path = args.meta_json or meta_path_auto
    depth_path = args.center_depth_path or depth_path_auto

    log.info(f"Using meta sidecar: {meta_path}")
    log.info(f"Using center depth sidecar: {depth_path}")

    meta = _load_meta(meta_path)
    fov_deg: float = float(meta["fov_deg"])
    viewcone_deg: float = float(meta["viewcone_deg"])
    camera_size: float = float(meta["camera_size"])
    invert_quilt: bool = bool(meta["invert_quilt"])
    order_from_meta: str = meta.get("order", args.order)

    # center tile index and seed color
    center_idx = _center_tile_index(cols, rows, order_from_meta)
    if center_idx < 0 or center_idx >= len(frames_rgb):
        raise ValueError(f"Computed center index {center_idx} out of range for {len(frames_rgb)} frames.")
    center_rgb = frames_rgb[center_idx]  # HxWx3 (uint8)

    # load center depth (16-bit PNG, 1.0 = NEAR) and map to meters
    if not os.path.exists(depth_path):
        raise FileNotFoundError(
            f"Missing center depth sidecar: {depth_path}\n"
            f"Expected alongside the quilt. Generate it in the quilt renderer."
        )
    depth_u16 = np.array(Image.open(depth_path), dtype=np.uint16)
    if depth_u16.ndim == 3:  # tolerate accidental RGB16 saves
        depth_u16 = depth_u16[..., 0]
    depth01 = depth_u16.astype(np.float32) / 65535.0
    if depth01.shape != (args.height, args.width):
        depth01 = cv2.resize(depth01, (args.width, args.height), interpolation=cv2.INTER_AREA)
    depth_m = _depth01_to_meters(depth01, near_m=args.depth_near_m, far_m=args.depth_far_m)

    # intrinsics + per-view poses
    K = _compute_intrinsics(args.width, args.height, fov_deg)
    w2c_all = _lkg_w2c_list(cols, rows, fov_deg, viewcone_deg, camera_size, invert_quilt, order_from_meta)
    w2c_center = w2c_all[center_idx]

    # --- build warps/masks by reprojecting center (RGBD) into every view ---
    with _timed("Build conditioning (warps + masks from center RGBD)"):
        warps = []
        masks = []
        for k in range(len(frames_rgb)):
            w_chw, m_1hw = _warp_center_to_view(center_rgb, depth_m, K, w2c_center, w2c_all[k], device)
            warps.append((w_chw * 2.0 - 1.0).unsqueeze(0))  # 1x3xHxW in [-1,1]
            masks.append(m_1hw.unsqueeze(0))                # 1x1xHxW

        cond_b1tchw = torch.cat([w.unsqueeze(0) for w in warps], dim=1)    # 1xT x3xHxW
        masks_b1t1hw = torch.cat([m.unsqueeze(0) for m in masks], dim=1)   # 1xT x1xHxW

        # Add N axis = 1 → shapes B x T x N x C x H x W and B x T x N x 1 x H x W
        imgs_6d = cond_b1tchw.unsqueeze(2)
        masks_6d = masks_b1t1hw.unsqueeze(2)

        B, T, _, C, H, W = imgs_6d.shape
        log.info(f"warps: {tuple(imgs_6d.shape)} range[{imgs_6d.min().item():.3f},{imgs_6d.max().item():.3f}]")
        log.info(f"masks: {tuple(masks_6d.shape)} mean={masks_6d.float().mean().item():.3f}")

    # optional: save conditioning preview
    if args.save_conditioning_video:
        try:
            cond_NHWC = ((cond_b1tchw[0].permute(0, 2, 3, 1).float() * 0.5 + 0.5) * 255.0).clamp(0, 255).byte().cpu().numpy()
            cond_path = os.path.join(args.video_save_folder, args.conditioning_video_name)
            with _timed("Save conditioning video"):
                save_video(video=cond_NHWC, fps=args.fps, H=args.height, W=args.width, video_save_quality=5, video_save_path=cond_path)
            log.info(f"Saved conditioning video to {cond_path}")
        except Exception as e:
            log.exception(f"Failed to save conditioning video (non-fatal): {e}")

    # --- init pipeline ---
    with _timed("Init Gen3cPipeline"):
        pipeline = Gen3cPipeline(
            inference_type="video2world",
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name="Gen3C-Cosmos-7B",
            prompt_upsampler_dir=args.prompt_upsampler_dir,
            enable_prompt_upsampler=not args.disable_prompt_upsampler,
            offload_network=args.offload_diffusion_transformer,
            offload_tokenizer=args.offload_tokenizer,
            offload_text_encoder_model=args.offload_text_encoder_model,
            offload_prompt_upsampler=args.offload_prompt_upsampler,
            offload_guardrail_models=args.offload_guardrail_models,
            disable_guardrail=args.disable_guardrail,
            disable_prompt_encoder=args.disable_prompt_encoder,
            guidance=args.guidance,
            num_steps=args.num_steps,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_video_frames=cond_b1tchw.shape[1],
            seed=args.seed,
        )

    sample_n_frames = int(pipeline.model.chunk_size)
    T = int(cond_b1tchw.shape[1])
    log.info(f"Model chunk_size={sample_n_frames}, input_frames={T}")

    if args.strict_exact_length and T != sample_n_frames:
        raise ValueError(
            f"Input quilt has {T} frames but model chunk_size is {sample_n_frames}. "
            f"To mirror the single-image script, provide a quilt with cols*rows=={sample_n_frames}."
        )

    # --- write seed center image and generate ---
    seed_path = os.path.join(args.video_save_folder, f"{args.video_save_name}_seed_center.png")
    Image.fromarray(center_rgb).save(seed_path)

    video_out_list = []
    with _timed("Generate"):
        if T <= sample_n_frames:
            # Single chunk
            generated = pipeline.generate(
                prompt=args.prompt or "",
                image_path=seed_path,                # IMPORTANT: path string, like the single-image script
                negative_prompt=args.negative_prompt,
                rendered_warp_images=imgs_6d[:, :T],
                rendered_warp_masks=masks_6d[:, :T],
            )
            if generated is None:
                log.critical("Guardrail blocked generation; aborting.")
                raise RuntimeError("Generation blocked by guardrail.")
            video_chunk, _prompt_text = generated
            video_out_list.append(video_chunk)
        else:
            # Multi-chunk with 1-frame overlap; reseed from last output frame (write to file each time)
            start = 0
            current_seed_path = seed_path
            chunk_idx = 0
            while start < T:
                end = min(start + sample_n_frames, T)
                log.info(f"[chunk {chunk_idx}] frames {start}:{end}")

                generated = pipeline.generate(
                    prompt=args.prompt or "",
                    image_path=current_seed_path,
                    negative_prompt=args.negative_prompt,
                    rendered_warp_images=imgs_6d[:, start:end],
                    rendered_warp_masks=masks_6d[:, start:end],
                )
                if generated is None:
                    log.critical("Guardrail blocked generation; aborting this item.")
                    break
                video_chunk, _prompt_text = generated

                if start == 0:
                    video_out_list.append(video_chunk)
                else:
                    video_out_list.append(video_chunk[1:])  # drop overlap frame

                # prepare next seed from the last output frame (write PNG)
                last_frame = video_chunk[-1]  # HxWxC uint8
                current_seed_path = os.path.join(
                    args.video_save_folder, f"{args.video_save_name}_seed_{chunk_idx+1:03d}.png"
                )
                Image.fromarray(last_frame).save(current_seed_path)

                # advance with overlap
                start = end - 1
                chunk_idx += 1

    if not video_out_list:
        raise RuntimeError("No output frames produced.")

    video_array = np.concatenate(video_out_list, axis=0)  # T H W C (uint8)
    out_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.mp4")
    with _timed("Save output video"):
        save_video(video=video_array, fps=args.fps, H=args.height, W=args.width, video_save_quality=5, video_save_path=out_path)
    log.info(f"Saved video to {out_path}")

    # Optional manifest
    if args.save_run_manifest:
        manifest = {
            "input": {
                "quilt_path": os.path.abspath(args.quilt_path),
                "meta_json": os.path.abspath(meta_path),
                "center_depth_path": os.path.abspath(depth_path),
                "cols": cols,
                "rows": rows,
                "order": args.order,
            },
            "meta": {
                "fov_deg": fov_deg,
                "viewcone_deg": viewcone_deg,
                "camera_size": camera_size,
                "invert_quilt": invert_quilt,
                "order_from_meta": order_from_meta,
            },
            "model": {
                "height": args.height,
                "width": args.width,
                "fps": args.fps,
                "guidance": args.guidance,
                "num_steps": args.num_steps,
                "seed": args.seed,
                "chunk_size": sample_n_frames,
                "frames_in": T,
            },
            "output": {
                "video_path": os.path.abspath(out_path),
                "conditioning_video": (os.path.abspath(os.path.join(args.video_save_folder, args.conditioning_video_name))
                                       if args.save_conditioning_video else None),
            },
            "device": str(device),
            "time_sec_total": float(time.perf_counter() - t_total_start),
        }
        man_name = args.run_manifest_name or f"{args.video_save_name}.manifest.json"
        man_path = os.path.join(args.video_save_folder, man_name)
        try:
            with open(man_path, "w") as f:
                json.dump(manifest, f, indent=2)
            log.info(f"Wrote run manifest: {man_path}")
        except Exception as e:
            log.exception(f"Failed to write run manifest: {e}")

    log.info(f"Overall elapsed: {(time.perf_counter() - t_total_start):.3f}s")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    demo(args)
