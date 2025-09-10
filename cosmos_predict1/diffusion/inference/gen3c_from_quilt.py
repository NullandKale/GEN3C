# gen3c_from_quilt.py
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
from contextlib import contextmanager

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

def _frames_to_b1tchw_minus1_1(frames_rgb_uint8: list[np.ndarray], height: int, width: int, device: torch.device) -> torch.Tensor:
    outs = []
    for f in frames_rgb_uint8:
        if f.shape[0] != height or f.shape[1] != width:
            f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(f).to(device=device, dtype=torch.float32) / 255.0
        t = t.permute(2, 0, 1)  # C,H,W
        outs.append(t)
    if not outs:
        raise ValueError("No frames extracted from quilt.")
    stack = torch.stack(outs, dim=0)   # T,C,H,W
    stack = stack.unsqueeze(0)         # 1,T,C,H,W
    stack = stack * 2.0 - 1.0          # [-1,1]
    return stack

def _ones_masks_like_images_b1tchw(imgs_b1tchw: torch.Tensor) -> torch.Tensor:
    B, T, C, H, W = imgs_b1tchw.shape
    masks = torch.ones((B, T, 1, H, W), dtype=torch.float32, device=imgs_b1tchw.device)
    return masks

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
    tile_w = (work_w - (grid_cols - 1) * gap_x) // grid_cols if grid_cols > 1 else work_w
    tile_h = (work_h - (grid_rows - 1) * gap_y) // grid_rows if grid_rows > 1 else work_h
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

def _snap_hw(height: int, width: int) -> tuple[int, int]:
    """
    GEN3C/Cosmos-7B prefers 1280x704 (16:9, multiples of 64). Fall back to 832x480 for '480p' if smaller.
    If arbitrary sizes are given, snap both dims to nearest multiple of 64 while keeping aspect close.
    """
    # already the preferred size
    if (width, height) == (1280, 704):
        return height, width
    # small preset
    if (width, height) == (832, 480):
        return height, width

    # try to keep 16:9 while snapping to multiples of 64
    def _snap64(x: int) -> int:
        return int(round(x / 64.0)) * 64

    w64 = _snap64(width)
    h64 = _snap64(height)
    # nudge towards the canonical pair if close
    if abs((w64 / max(h64,1)) - (16/9)) < 0.02:
        if w64 >= 1280:
            return 704, 1280
        if w64 >= 832:
            return 480, 832
    # last resort: just use 1280x704 if the input is roughly widescreen
    ar = width / max(height, 1)
    if 1.6 <= ar <= 1.85:
        return 704, 1280
    # otherwise stick with snapped 64s
    return h64, w64

def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GEN3C from quilt image: unpack grid -> fixed-length conditioning video -> generate (mirrors single-image process).")
    p.add_argument("--quilt_path", type=str, required=True, help="Path to the quilt image with frames arranged in a grid (supports _qs{C}x{R}a{AR} in the filename).")
    p.add_argument("--grid_cols", type=int, default=0, help="Number of columns in the quilt grid; if 0, parsed from filename.")
    p.add_argument("--grid_rows", type=int, default=0, help="Number of rows in the quilt grid; if 0, parsed from filename.")
    p.add_argument("--total_frames", type=int, default=None, help="Optional cap on frames to read; default uses all tiles.")
    p.add_argument("--order", type=str, choices=["quilt", "row-major", "col-major"], default="quilt", help="Traversal order for tiles. 'quilt' matches Looking Glass (bottom-left origin).")
    p.add_argument("--left", type=int, default=0, help="Left margin to skip before tiles.")
    p.add_argument("--top", type=int, default=0, help="Top margin to skip before tiles.")
    p.add_argument("--right", type=int, default=0, help="Right margin to skip after tiles.")
    p.add_argument("--bottom", type=int, default=0, help="Bottom margin to skip after tiles.")
    p.add_argument("--gap_x", type=int, default=0, help="Horizontal gap (pixels) between tiles.")
    p.add_argument("--gap_y", type=int, default=0, help="Vertical gap (pixels) between tiles.")
    p.add_argument("--checkpoint_dir", type=str, required=True, help="Path to gen3c checkpoint directory (contains Gen3C-Cosmos-7B).")
    p.add_argument("--video_save_folder", type=str, default="outputs", help="Folder for generated outputs.")
    p.add_argument("--video_save_name", type=str, default="gen3c_from_quilt", help="Base name for the output MP4.")
    p.add_argument("--height", type=int, default=704, help="Target H for generation and conditioning frames (snap to multiples of 64).")
    p.add_argument("--width", type=int, default=1280, help="Target W for generation and conditioning frames (snap to multiples of 64).")
    p.add_argument("--fps", type=int, default=24, help="FPS for output video.")
    p.add_argument("--guidance", type=float, default=1.0, help="CFG guidance.")
    p.add_argument("--num_steps", type=int, default=25, help="Diffusion steps.")
    p.add_argument("--seed", type=int, default=1234, help="Random seed.")
    p.add_argument("--disable_guardrail", action="store_true", help="Disable guardrails.")
    p.add_argument("--disable_prompt_upsampler", action="store_true", help="Disable prompt upsampler.")
    p.add_argument("--prompt_upsampler_dir", type=str, default="Pixtral-12B", help="Prompt upsampler weights directory relative to checkpoint_dir.")
    p.add_argument("--negative_prompt", type=str, default=None, help="Optional negative prompt.")
    p.add_argument("--prompt", type=str, default="", help="Text prompt (optional).")
    p.add_argument("--offload_diffusion_transformer", action="store_true", help="Offload diffusion transformer.")
    p.add_argument("--offload_tokenizer", action="store_true", help="Offload tokenizer.")
    p.add_argument("--offload_text_encoder_model", action="store_true", help="Offload text encoder model.")
    p.add_argument("--offload_prompt_upsampler", action="store_true", help="Offload prompt upsampler.")
    p.add_argument("--offload_guardrail_models", action="store_true", help="Offload guardrail models.")
    p.add_argument("--disable_prompt_encoder", action="store_true", help="Disable prompt encoder.")
    p.add_argument("--save_conditioning_video", action="store_true", help="If set, write the unpacked conditioning video.")
    p.add_argument("--conditioning_video_name", type=str, default="conditioning_from_quilt.mp4", help="Filename for conditioning video.")
    p.add_argument("--strict_exact_length", action="store_true", help="Require that quilt frames exactly equal model chunk_size (recommended to mimic single-image script).")
    return p

def _resolve_grid_args(quilt_path: str, cols: int, rows: int) -> tuple[int, int]:
    if cols > 0 and rows > 0:
        return cols, rows
    c2, r2, _ = _parse_quilt_meta_from_name(quilt_path)
    if c2 is None or r2 is None:
        raise ValueError("grid_cols/grid_rows not provided and filename does not encode _qs{C}x{R}.")
    log.info(f"Parsed grid from filename: cols={c2}, rows={r2}")
    return c2, r2

def demo(args: argparse.Namespace) -> None:
    t_total_start = time.perf_counter()
    misc.set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} (CUDA={torch.cuda.is_available()})")
    _ensure_dir(args.video_save_folder)

    cols, rows = _resolve_grid_args(args.quilt_path, args.grid_cols, args.grid_rows)

    # Enforce safe resolution (prevents 128 vs 160 latent mismatch)
    H_in, W_in = int(args.height), int(args.width)
    H_use, W_use = _snap_hw(H_in, W_in)
    if (H_use, W_use) != (H_in, W_in):
        log.info(f"Snapped resolution from {W_in}x{H_in} -> {W_use}x{H_use} (multiples of 64; 16:9 favored).")

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

    with _timed("Build conditioning tensor"):
        cond_b1tchw = _frames_to_b1tchw_minus1_1(frames_rgb, H_use, W_use, device)
        masks_b1t1hw = _ones_masks_like_images_b1tchw(cond_b1tchw)
        B, T, C, H, W = cond_b1tchw.shape
        # For internal model helper paths that expect a 6D (B,T,N,C,H,W), we can expand N=1 as needed.
        cond_b1ntchw = cond_b1tchw.unsqueeze(2)  # B,T,1,C,H,W
        masks_b1n1hw = masks_b1t1hw.unsqueeze(2) # B,T,1,1,H,W
        log.info(f"Conditioning (B,T,C,H,W): {tuple(cond_b1tchw.shape)}; with N-axis: {tuple(cond_b1ntchw.shape)}")
        log.info(f"Masks       (B,T,1,H,W): {tuple(masks_b1t1hw.shape)}; with N-axis: {tuple(masks_b1n1hw.shape)}")

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
            height=H_use,
            width=W_use,
            fps=args.fps,
            num_video_frames=T,
            seed=args.seed,
        )

    sample_n_frames = int(pipeline.model.chunk_size)
    log.info(f"Model chunk_size={sample_n_frames}, input_frames={T}")

    if args.strict_exact_length and T != sample_n_frames:
        raise ValueError(f"Input quilt has {T} frames but model chunk_size is {sample_n_frames}. To mirror the single-image script, provide a quilt with cols*rows=={sample_n_frames} (e.g., _qs11x11...).")

    if args.save_conditioning_video:
        try:
            cond_NHWC = ((cond_b1tchw[0].permute(0, 2, 3, 1).float() * 0.5 + 0.5) * 255.0).clamp(0, 255).byte().cpu().numpy()
            cond_path = os.path.join(args.video_save_folder, args.conditioning_video_name)
            with _timed("Save conditioning video"):
                save_video(video=cond_NHWC, fps=args.fps, H=H_use, W=W_use, video_save_quality=5, video_save_path=cond_path)
            log.info(f"Saved conditioning video to {cond_path}")
        except Exception as e:
            log.exception(f"Failed to save conditioning video (non-fatal): {e}")

    video_out_list = []
    with _timed("Generate"):
        if T <= sample_n_frames:
            imgs = cond_b1tchw[:, :T]
            masks = masks_b1t1hw[:, :T]
            img0_bcthw = imgs[:, 0:1].permute(0, 2, 1, 3, 4)  # B,C,T,H,W with T=1
            generated = pipeline.generate(
                prompt=args.prompt or "",
                image_path=img0_bcthw,
                negative_prompt=args.negative_prompt,
                rendered_warp_images=imgs,
                rendered_warp_masks=masks,
            )
            if generated is None:
                log.critical("Guardrail blocked generation; aborting.")
                raise RuntimeError("Generation blocked by guardrail.")
            video_chunk, prompt_text = generated
            video_out_list.append(video_chunk)
        else:
            start = 0
            last_seed_frame_bcthw = None
            while start < T:
                end = min(start + sample_n_frames, T)
                chunk_imgs = cond_b1tchw[:, start:end]
                chunk_masks = masks_b1t1hw[:, start:end]
                if last_seed_frame_bcthw is None:
                    img0_bcthw = chunk_imgs[:, 0:1].permute(0, 2, 1, 3, 4)
                else:
                    img0_bcthw = last_seed_frame_bcthw
                generated = pipeline.generate(
                    prompt=args.prompt or "",
                    image_path=img0_bcthw,
                    negative_prompt=args.negative_prompt,
                    rendered_warp_images=chunk_imgs,
                    rendered_warp_masks=chunk_masks,
                )
                if generated is None:
                    log.critical("Guardrail blocked generation; aborting this item.")
                    break
                video_chunk, prompt_text = generated
                if start == 0:
                    video_out_list.append(video_chunk)
                else:
                    video_out_list.append(video_chunk[1:])
                last_frame = torch.from_numpy(video_chunk[-1]).to(device=device, dtype=torch.float32) / 255.0
                last_chw = last_frame.permute(2, 0, 1)
                last_seed_frame_bcthw = (last_chw.unsqueeze(0).unsqueeze(2) * 2.0 - 1.0)
                start = end - 1

    if not video_out_list:
        raise RuntimeError("No output frames produced.")
    video_array = np.concatenate(video_out_list, axis=0)
    out_path = os.path.join(args.video_save_folder, f"{args.video_save_name}.mp4")
    with _timed("Save output video"):
        save_video(video=video_array, fps=args.fps, H=H_use, W=W_use, video_save_quality=5, video_save_path=out_path)
    log.info(f"Saved video to {out_path}")
    log.info(f"Overall elapsed: {(time.perf_counter() - t_total_start):.3f}s")

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    demo(args)
