# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import cv2
from moge.model.v1 import MoGeModel
import torch
import numpy as np
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
    check_input_frames,
)
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
import torch.nn.functional as F
import time
from contextlib import contextmanager

torch.enable_grad(False)

# ----------------------------
# Utilities
# ----------------------------
@contextmanager
def _timed(section: str):
    t0 = time.perf_counter()
    log.info(f"[TIMER] {section} | start")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log.info(f"[TIMER] {section} | end: {dt:.3f}s")

def _now_perf() -> float:
    return time.perf_counter()

def _ensure_dir(d: str | None):
    if d:
        os.makedirs(d, exist_ok=True)

def _save_mask_png(mask_hw_bool: torch.Tensor, out_path: str):
    """Save a binary mask (H, W) as 8-bit PNG (0 or 255)."""
    mask_u8 = (mask_hw_bool.to(torch.uint8) * 255).cpu().numpy()
    cv2.imwrite(out_path, mask_u8)

def _save_depth_png16(depth_hw: torch.Tensor,
                      valid_mask_hw: torch.Tensor | None,
                      out_path: str,
                      q_lo: float = 0.01,
                      q_hi: float = 0.99):
    """
    Save depth (H, W, float32) as a 16-bit single-channel PNG using robust
    percentile normalization over valid pixels.
    """
    d = depth_hw
    if valid_mask_hw is None:
        valid_mask_hw = torch.isfinite(d) & (d < 9.99e3)

    if not torch.any(valid_mask_hw):
        # Fallback: zeros
        arr = torch.zeros_like(d, dtype=torch.uint16).cpu().numpy()
        cv2.imwrite(out_path, arr)
        return

    vals = d[valid_mask_hw]
    lo = torch.quantile(vals, torch.tensor(q_lo, device=d.device))
    hi = torch.quantile(vals, torch.tensor(q_hi, device=d.device))
    scale = 65535.0 / max((hi - lo).item(), 1e-6)
    d16 = ((d - lo) * scale).clamp(0, 65535).to(torch.uint16).cpu().numpy()
    cv2.imwrite(out_path, d16)

# ----------------------------
# Args
# ----------------------------
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image-to-video (video-to-world) demo script")

    # Common project args
    add_common_arguments(parser)

    parser.add_argument("--prompt_upsampler_dir", type=str, default="Pixtral-12B",
                        help="Prompt upsampler weights directory relative to checkpoint_dir")
    parser.add_argument("--input_image_path", type=str,
                        help="Input image path for generating a single video")

    parser.add_argument("--trajectory", type=str,
                        choices=["left", "right", "up", "down",
                                 "zoom_in", "zoom_out", "clockwise", "counterclockwise", "none"],
                        default="left",
                        help="Camera path for synthetic motion.")
    parser.add_argument("--camera_rotation", type=str,
                        choices=["center_facing", "no_rotation", "trajectory_aligned"],
                        default="center_facing",
                        help="Rotation behavior during movement.")
    parser.add_argument("--movement_distance", type=float, default=0.3,
                        help="Distance of the camera from the scene center.")
    parser.add_argument("--noise_aug_strength", type=float, default=0.0,
                        help="Noise augmentation on warped frames.")
    parser.add_argument("--save_buffer", action="store_true",
                        help="If set, show rendered warp buffers side-by-side with output video.")
    parser.add_argument("--filter_points_threshold", type=float, default=0.05,
                        help="Filter threshold for point continuity in warps.")
    parser.add_argument("--foreground_masking", action="store_true",
                        help="Use foreground masking for warps.")

    # New outputs
    parser.add_argument("--save_depth_dir", type=str, default=None,
                        help="If set, saves depth PNG16 frames here (e.g., outputs/depth).")
    parser.add_argument("--save_mask_dir", type=str, default=None,
                        help="If set, saves mask PNG frames here (e.g., outputs/mask).")
    parser.add_argument("--save_conditioning_video", action="store_true",
                        help="If set, saves the camera-warped conditioning sequence as MP4.")
    parser.add_argument("--conditioning_video_name", type=str, default="input_conditioning.mp4",
                        help="Filename for the conditioning MP4 (in video_save_folder).")

    return parser

def parse_arguments() -> argparse.Namespace:
    parser = create_parser()
    return parser.parse_args()

def validate_args(args):
    assert args.num_video_frames is not None, "num_video_frames must be provided"
    assert (args.num_video_frames - 1) % 120 == 0, "num_video_frames must be 121, 241, 361, ... (N*120+1)"

# ----------------------------
# Depth/Intrinsics via MoGe (no DAD)
# ----------------------------
def _predict_moge_depth_intrinsics(
    current_image_path_or_rgb: str | np.ndarray,
    target_h: int,
    target_w: int,
    device: torch.device,
    moge_model: MoGeModel
):
    """
    Run MoGe to get depth, intrinsics, mask and a resized/normalized image tensor.
    Returns:
        - image_b1chw_float in [-1, 1]
        - depth_b11hw (meters-ish, clamped)
        - mask_b11hw (bool-like)
        - initial_w2c_b144 (eye)
        - intrinsics_b133 (pixel units, resized)
    """
    # Load image
    if isinstance(current_image_path_or_rgb, str):
        input_bgr = cv2.imread(current_image_path_or_rgb)
        if input_bgr is None:
            raise FileNotFoundError(f"Input image not found: {current_image_path_or_rgb}")
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
    else:
        input_rgb = current_image_path_or_rgb
    del current_image_path_or_rgb

    # MoGe expects 1280x720 internally
    depth_pred_h, depth_pred_w = 720, 1280

    # Prepare tensor for MoGe
    img_resized = cv2.resize(input_rgb, (depth_pred_w, depth_pred_h))
    img_chw_0_1 = torch.tensor(img_resized / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

    with _timed("MoGe infer"):
        moge_out = moge_model.infer(img_chw_0_1)

    depth_hw_full = moge_out["depth"]                  # (H, W)
    intrinsics_33_full_norm = moge_out["intrinsics"]   # normalized
    mask_hw_full = moge_out["mask"]                    # 0/1

    # Replace invalid with large sentinel
    depth_hw_full = torch.where(mask_hw_full == 0, torch.tensor(1000.0, device=depth_hw_full.device), depth_hw_full)

    # Convert intrinsics to pixel units at 1280x720
    intr_pix = intrinsics_33_full_norm.clone()
    intr_pix[0, 0] *= depth_pred_w
    intr_pix[1, 1] *= depth_pred_h
    intr_pix[0, 2] *= depth_pred_w
    intr_pix[1, 2] *= depth_pred_h

    # Resize to target resolution
    height_scale = target_h / depth_pred_h
    width_scale  = target_w / depth_pred_w

    depth_hw = F.interpolate(depth_hw_full[None, None], size=(target_h, target_w),
                             mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    mask_hw  = F.interpolate(mask_hw_full[None, None].float(), size=(target_h, target_w),
                             mode='nearest').squeeze(0).squeeze(0).bool()

    img_chw_target = F.interpolate(img_chw_0_1[None], size=(target_h, target_w),
                                   mode='bilinear', align_corners=False).squeeze(0)
    image_b1chw_float = img_chw_target[None, None] * 2 - 1  # [-1, 1], shape (B=1,1,C,H,W)

    intrinsics_33 = intr_pix.clone()
    intrinsics_33[1, 1] *= height_scale
    intrinsics_33[1, 2] *= height_scale
    intrinsics_33[0, 0] *= width_scale
    intrinsics_33[0, 2] *= width_scale

    depth_b11hw = depth_hw[None, None, None]
    depth_b11hw = torch.nan_to_num(depth_b11hw, nan=1e4).clamp_(0, 1e4)
    mask_b11hw  = mask_hw[None, None, None]
    intr_b133   = intrinsics_33[None, None]
    w2c_b144    = torch.eye(4, dtype=torch.float32, device=device)[None, None]

    return image_b1chw_float, depth_b11hw, mask_b11hw, w2c_b144, intr_b133

def _predict_moge_depth_from_image_tensor(
    image_tensor_chw_0_1: torch.Tensor,  # (C,H,W), [0,1]
    moge_model: MoGeModel
):
    """Run MoGe on a single RGB frame tensor -> (depth_11hw, mask_11hw)."""
    with _timed("MoGe infer (AR frame)"):
        out = moge_model.infer(image_tensor_chw_0_1)
    depth_hw = out["depth"]
    mask_hw  = out["mask"].bool()
    depth_11 = torch.nan_to_num(depth_hw[None, None], nan=1e4).clamp_(0, 1e4)
    mask_11  = mask_hw[None, None]
    # Put very large sentinel depth on invalid pixels to avoid artifacts in later stages
    depth_11 = torch.where(mask_11 == 0, torch.tensor(1000.0, device=depth_11.device), depth_11)
    return depth_11, mask_11

# ----------------------------
# Main
# ----------------------------
def demo(args):
    """
    End-to-end:
      1) MoGe predicts depth+intrinsics+mask for the input image (no DAD).
      2) Cache3D buffers the initial frame for geometric warping along a camera trajectory.
      3) Render warped conditioning frames (the "input video" to diffusion).
      4) Gen3C synthesizes frames chunk-by-chunk; AR passes update Cache3D with latest frame's depth.
      5) Optionally save:
          - depth frames (PNG16),
          - mask frames (PNG),
          - conditioning sequence (MP4),
          - final video (MP4, optionally side-by-side with warps).
    """
    t_total_start = _now_perf()
    misc.set_random_seed(args.seed)
    validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Runtime device: {device}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            log.info(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    log.info(f"Args snapshot: {args}")

    # Pipeline
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
            num_video_frames=args.num_video_frames,
            seed=args.seed,
        )

    frame_buffer_max = pipeline.model.frame_buffer_max
    sample_n_frames  = pipeline.model.chunk_size
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Load MoGe
    with _timed("Load MoGe"):
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)

    # Inputs
    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        prompts = [{"prompt": args.prompt or "", "visual_input": args.input_image_path}]

    # Output dirs
    os.makedirs(args.video_save_folder, exist_ok=True)
    _ensure_dir(args.save_depth_dir)
    _ensure_dir(args.save_mask_dir)

    for i, input_dict in enumerate(prompts):
        t_item_start = _now_perf()
        current_prompt = input_dict.get("prompt", "")
        current_image_path = input_dict.get("visual_input", None)
        if current_image_path is None:
            log.critical("Visual input is missing, skipping.")
            continue

        if not check_input_frames(current_image_path, 1):
            log.critical(f"Input image {current_image_path} is not valid, skipping.")
            continue

        log.info(f"Item {i}: input={current_image_path}, movement={args.movement_distance}, "
                 f"frames={args.num_video_frames}, chunk={sample_n_frames}, guidance={args.guidance}")

        # MoGe depth/intrinsics/mask + initial cache
        with _timed("Depth+intrinsics init (MoGe)"):
            (moge_image_b1chw_float,
             moge_depth_b11hw,
             moge_mask_b11hw,
             moge_initial_w2c_b144,
             moge_intrinsics_b133) = _predict_moge_depth_intrinsics(
                current_image_path, args.height, args.width, device, moge_model
            )

        # Optional save of the very first depth/mask
        if args.save_depth_dir:
            _save_depth_png16(
                moge_depth_b11hw[0, 0, 0],  # (H,W)
                moge_mask_b11hw[0, 0, 0],
                os.path.join(args.save_depth_dir, f"{args.video_save_name}_depth_{0:04d}.png"),
            )
        if args.save_mask_dir:
            _save_mask_png(
                moge_mask_b11hw[0, 0, 0],
                os.path.join(args.save_mask_dir, f"{args.video_save_name}_mask_{0:04d}.png"),
            )

        with _timed("Cache3D init"):
            cache = Cache3D_Buffer(
                frame_buffer_max=frame_buffer_max,
                generator=generator,
                noise_aug_strength=args.noise_aug_strength,
                input_image=moge_image_b1chw_float[:, 0].clone(),
                input_depth=moge_depth_b11hw[:, 0],
                # input_mask=moge_mask_b11hw[:, 0],  # optional, kept disabled to match earlier behavior
                input_w2c=moge_initial_w2c_b144[:, 0],
                input_intrinsics=moge_intrinsics_b133[:, 0],
                filter_points_threshold=args.filter_points_threshold,
                foreground_masking=args.foreground_masking,
            )

        # Trajectory
        initial_cam_w2c_for_traj = moge_initial_w2c_b144[0, 0]
        initial_cam_intrinsics_for_traj = moge_intrinsics_b133[0, 0]
        with _timed("Trajectory generation"):
            generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                trajectory_type=args.trajectory,
                initial_w2c=initial_cam_w2c_for_traj,
                initial_intrinsics=initial_cam_intrinsics_for_traj,
                num_frames=args.num_video_frames,
                movement_distance=args.movement_distance,
                camera_rotation=args.camera_rotation,
                center_depth=1.0,
                device=device.type,
            )

        # Render the first conditioning chunk
        with _timed(f"Render cache [0:{sample_n_frames}]"):
            rendered_warp_images, rendered_warp_masks = cache.render_cache(
                generated_w2cs[:, 0:sample_n_frames],
                generated_intrinsics[:, 0:sample_n_frames],
            )

        # For optional conditioning video: keep a running list of warp chunks
        conditioning_chunks = []
        if args.save_conditioning_video:
            conditioning_chunks.append(rendered_warp_images.clone().cpu())  # (1, N, 3, H, W)

        # Optionally also show buffers side-by-side in final video
        all_rendered_warps_for_sbs = []
        if args.save_buffer:
            all_rendered_warps_for_sbs.append(rendered_warp_images.clone().cpu())

        # Generate first chunk
        with _timed(f"Pipeline.generate [0:{sample_n_frames}]"):
            generated_output = pipeline.generate(
                prompt=current_prompt,
                image_path=current_image_path,  # path string for first call
                negative_prompt=args.negative_prompt,
                rendered_warp_images=rendered_warp_images,
                rendered_warp_masks=rendered_warp_masks,
            )
        if generated_output is None:
            log.critical("Guardrail blocked generation.")
            continue
        video, prompt_text = generated_output  # video: (T, H, W, 3) uint8

        # Autoregressive passes
        num_ar_iterations = (generated_w2cs.shape[1] - 1) // (sample_n_frames - 1)
        for num_iter in range(1, num_ar_iterations):
            start_frame_idx = num_iter * (sample_n_frames - 1)
            end_frame_idx = start_frame_idx + sample_n_frames
            log.info(f"AR pass {num_iter}/{num_ar_iterations - 1}: frames [{start_frame_idx}:{end_frame_idx}]")

            # Use last synthesized frame to bootstrap new depth
            last_frame_hwc_0_255 = torch.tensor(video[-1], device=device)        # (H,W,3) uint8
            frame_chw_0_1 = last_frame_hwc_0_255.permute(2, 0, 1).float() / 255.0

            with _timed("MoGe depth predict (AR)"):
                pred_depth_11, pred_mask_11 = _predict_moge_depth_from_image_tensor(frame_chw_0_1, moge_model)

            # Optional save AR depth/mask (use the timeline index where this frame lands)
            if args.save_depth_dir:
                _save_depth_png16(
                    pred_depth_11[0, 0], pred_mask_11[0, 0],
                    os.path.join(args.save_depth_dir, f"{args.video_save_name}_depth_{start_frame_idx:04d}.png")
                )
            if args.save_mask_dir:
                _save_mask_png(
                    pred_mask_11[0, 0],
                    os.path.join(args.save_mask_dir, f"{args.video_save_name}_mask_{start_frame_idx:04d}.png")
                )

            # Update cache with this latest frame's geometry
            with _timed("Cache3D update"):
                cache.update_cache(
                    new_image=frame_chw_0_1[None] * 2 - 1,  # [-1,1], (1,C,H,W)
                    new_depth=pred_depth_11,
                    # new_mask=pred_mask_11,  # keep disabled (consistent with above)
                    new_w2c=generated_w2cs[:, start_frame_idx],
                    new_intrinsics=generated_intrinsics[:, start_frame_idx],
                )

            current_w2cs = generated_w2cs[:, start_frame_idx:end_frame_idx]
            current_intr = generated_intrinsics[:, start_frame_idx:end_frame_idx]
            with _timed(f"Render cache [{start_frame_idx}:{end_frame_idx}]"):
                rendered_warp_images, rendered_warp_masks = cache.render_cache(current_w2cs, current_intr)

            if args.save_conditioning_video:
                conditioning_chunks.append(rendered_warp_images[:, 1:].clone().cpu())  # drop overlap

            if args.save_buffer:
                all_rendered_warps_for_sbs.append(rendered_warp_images[:, 1:].clone().cpu())

            # Generate next chunk (now passing a tensor image)
            frame_bcthw_minus1_1 = frame_chw_0_1[None, None] * 2 - 1  # (1,1,C,H,W)
            with _timed(f"Pipeline.generate [{start_frame_idx}:{end_frame_idx}]"):
                generated_output = pipeline.generate(
                    prompt=current_prompt,
                    image_path=frame_bcthw_minus1_1,
                    negative_prompt=args.negative_prompt,
                    rendered_warp_images=rendered_warp_images,
                    rendered_warp_masks=rendered_warp_masks,
                )
            video_new, prompt_text = generated_output
            video = np.concatenate([video, video_new[1:]], axis=0)  # append, skip duplicate

        # Compose final video, optionally side-by-side with warp buffers
        final_video_to_save = video
        final_width = args.width

        if args.save_buffer and all_rendered_warps_for_sbs:
            # Stack all rendered warp chunks to same N and concat horizontally with the output
            squeezed = [t.squeeze(0) for t in all_rendered_warps_for_sbs]  # [(N,C,H,W), ...]
            n_max = max(t.shape[0] for t in squeezed)

            padded_list = []
            for t in squeezed:
                # pad along time N
                pad_n = n_max - t.shape[0]
                pad_spec = (0, 0, 0, 0, 0, 0, 0, pad_n)  # (W,C,H,N) reversed by F.pad order
                # Convert to (C,H,N,W) before pad for simplicity
                t_CHNW = t.permute(1, 2, 0, 3)
                t_pad = F.pad(t_CHNW, pad_spec, mode='constant', value=-1.0)
                padded_list.append(t_pad)

            # Cat along CHNW batch (i.e., rows), then unstack into a single wide image per frame
            cat_CHNW = torch.cat(padded_list, dim=0)                               # (C*k, H, N, W)
            # Back to (N,C,H,k*W)
            cat_NCHW = cat_CHNW.permute(2, 0, 1, 3)
            # [-1,1] -> [0,255]
            cat_NCHW = ((cat_NCHW * 0.5 + 0.5) * 255.0).clamp(0, 255).byte()
            cat_NHWC = cat_NCHW.permute(0, 2, 3, 1).cpu().numpy()                  # (N,H,W_all,C)
            final_video_to_save = np.concatenate([cat_NHWC, final_video_to_save], axis=2)
            final_width = final_video_to_save.shape[2]
            log.info(f"Concatenated with warp buffers. Final width = {final_width}")

        # Save final video
        video_save_path = os.path.join(
            args.video_save_folder,
            f"{i if args.batch_input_path else args.video_save_name}.mp4"
        )
        with _timed("Save video"):
            save_video(
                video=final_video_to_save,
                fps=args.fps,
                H=args.height,
                W=final_width,
                video_save_quality=5,
                video_save_path=video_save_path,
            )
        log.info(f"Saved video to {video_save_path}")

        # Save the conditioning ("input") video if requested.
        # IMPORTANT: This block logs and continues on ANY failure.
        if args.save_conditioning_video and conditioning_chunks:
            try:
                for idx, t in enumerate(conditioning_chunks):
                    log.info(f"[COND] chunk[{idx}] dtype={t.dtype} device={t.device} shape={tuple(t.shape)}")

                # Concatenate along time dimension; first chunk keeps all frames, others dropped overlap already
                cond = torch.cat(conditioning_chunks, dim=1)  # expected ~ (B, T, C, H, W)
                log.info(f"[COND] stitched shape before reshape: {tuple(cond.shape)}; dtype={cond.dtype}; device={cond.device}")

                # Robust reshape: collapse ALL leading dims into T, keep (C,H,W) as the last three dims.
                if cond.dim() < 4:
                    raise RuntimeError(f"Conditioning tensor has too few dims: {cond.dim()} (shape={tuple(cond.shape)})")

                C, H, W = int(cond.shape[-3]), int(cond.shape[-2]), int(cond.shape[-1])
                N = int(np.prod(cond.shape[:-3]))  # total frames after flattening all leading dims
                cond_NCHW = cond.reshape(N, C, H, W)
                log.info(f"[COND] after reshape to (N,C,H,W): {tuple(cond_NCHW.shape)}")

                cond_THWC = ((cond_NCHW.permute(0, 2, 3, 1).float() * 0.5 + 0.5)
                              * 255.0).clamp(0, 255).byte().cpu().numpy()
                log.info(f"[COND] final THWC array shape: {cond_THWC.shape}")

                cond_path = os.path.join(args.video_save_folder, args.conditioning_video_name)
                with _timed("Save conditioning video"):
                    save_video(
                        video=cond_THWC,
                        fps=args.fps,
                        H=args.height,
                        W=args.width,
                        video_save_quality=5,
                        video_save_path=cond_path,
                    )
                log.info(f"Saved conditioning video to {cond_path}")
            except Exception as e:
                # Do not fail the whole run if conditioning video save fails.
                log.exception(f"[COND] Failed to save conditioning video; continuing without it. Error: {e}")

        log.info(f"Item {i} total elapsed: {(time.perf_counter() - t_item_start):.3f}s")

    log.info(f"Overall elapsed: {(time.perf_counter() - t_total_start):.3f}s")

if __name__ == "__main__":
    args = parse_arguments()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    demo(args)
