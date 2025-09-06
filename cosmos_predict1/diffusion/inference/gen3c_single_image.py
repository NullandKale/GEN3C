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
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

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
        arr = torch.zeros_like(d, dtype=torch.uint16).cpu().numpy()
        cv2.imwrite(out_path, arr)
        return
    vals = d[valid_mask_hw]
    lo = torch.quantile(vals, torch.tensor(q_lo, device=d.device))
    hi = torch.quantile(vals, torch.tensor(q_hi, device=d.device))
    scale = 65535.0 / max((hi - lo).item(), 1e-6)
    d16 = ((d - lo) * scale).clamp(0, 65535).to(torch.uint16).cpu().numpy()
    cv2.imwrite(out_path, d16)

def _full_true_mask_11hw(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Return mask of ones with shape (1,1,H,W)."""
    return torch.ones((1,1,H,W), dtype=torch.bool, device=device)

def _full_true_mask_hw(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Return mask of ones with shape (H,W)."""
    return torch.ones((H,W), dtype=torch.bool, device=device)

# ----------------------------
# Args
# ----------------------------
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image-to-video (video-to-world) demo script")
    add_common_arguments(parser)
    parser.add_argument("--prompt_upsampler_dir", type=str, default="Pixtral-12B", help="Prompt upsampler weights directory relative to checkpoint_dir")
    parser.add_argument("--input_image_path", type=str, help="Input image path for generating a single video")
    parser.add_argument("--trajectory", type=str, choices=["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise", "none"], default="left", help="Camera path for synthetic motion.")
    parser.add_argument("--camera_rotation", type=str, choices=["center_facing", "no_rotation", "trajectory_aligned"], default="center_facing", help="Rotation behavior during movement.")
    parser.add_argument("--movement_distance", type=float, default=0.3, help="Distance of the camera from the scene center.")
    parser.add_argument("--noise_aug_strength", type=float, default=0.0, help="Noise augmentation on warped frames.")
    parser.add_argument("--save_buffer", action="store_true", help="If set, show rendered warp buffers side-by-side with output video.")
    parser.add_argument("--filter_points_threshold", type=float, default=0.05, help="Filter threshold for point continuity in warps.")
    parser.add_argument("--foreground_masking", action="store_true", help="Use foreground masking for warps.")
    parser.add_argument("--save_depth_dir", type=str, default=None, help="If set, saves depth PNG16 frames here (e.g., outputs/depth).")
    parser.add_argument("--save_mask_dir", type=str, default=None, help="If set, saves mask PNG frames here (e.g., outputs/mask).")
    parser.add_argument("--save_conditioning_video", action="store_true", help="If set, saves the camera-warped conditioning sequence as MP4.")
    parser.add_argument("--conditioning_video_name", type=str, default="input_conditioning.mp4", help="Filename for the conditioning MP4 (in video_save_folder).")
    parser.add_argument("--dad_model_id", type=str, default="xingyang1/Distill-Any-Depth-Large-hf", help="Hugging Face model id for Distill-Any-Depth.")
    return parser

def parse_arguments() -> argparse.Namespace:
    parser = create_parser()
    return parser.parse_args()

def validate_args(args):
    assert args.num_video_frames is not None, "num_video_frames must be provided"
    assert (args.num_video_frames - 1) % 120 == 0, "num_video_frames must be 121, 241, 361, ... (N*120+1)"

# ----------------------------
# DAD depth helpers
# ----------------------------
def _dad_predict_depth_hw_from_rgb_numpy(input_image_rgb: np.ndarray,
                                         target_h: int,
                                         target_w: int,
                                         device: torch.device,
                                         dad_processor: AutoImageProcessor,
                                         dad_model: AutoModelForDepthEstimation) -> torch.Tensor:
    """Run Distill-Any-Depth via Transformers, return float32 tensor (H, W) on device."""
    with _timed("DAD infer"):
        inputs = dad_processor(images=[input_image_rgb], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = dad_model(**inputs)
        post = dad_processor.post_process_depth_estimation(outputs, target_sizes=[(target_h, target_w)])
        depth_hw = post[0]["predicted_depth"].to(device).to(torch.float32)
    return depth_hw

def _rescale_depth_like_reference(dad_hw: torch.Tensor,
                                  ref_hw: torch.Tensor) -> torch.Tensor:
    """Rescale DAD depth to match reference depth distribution using 5–95% percentiles."""
    device = dad_hw.device
    eps = torch.tensor(1e-6, device=device, dtype=torch.float32)
    q_ref = torch.quantile(ref_hw.reshape(-1), torch.tensor([0.05, 0.95], device=device))
    q_dad = torch.quantile(dad_hw.reshape(-1), torch.tensor([0.05, 0.95], device=device))
    dad_norm = (dad_hw - q_dad[0]) / (q_dad[1] - q_dad[0] + eps)
    dad_scaled = dad_norm * (q_ref[1] - q_ref[0]) + q_ref[0]
    dad_scaled = torch.nan_to_num(dad_scaled, nan=1e4).clamp_(0.0, 1e4)
    return dad_scaled

# ----------------------------
# Depth/Intrinsics via MoGe + DAD replacement, mask = all white
# ----------------------------
def _predict_initial_image_depth_intrinsics_with_dad(
    current_image_path_or_rgb: str | np.ndarray,
    target_h: int,
    target_w: int,
    device: torch.device,
    moge_model: MoGeModel,
    dad_processor: AutoImageProcessor,
    dad_model: AutoModelForDepthEstimation
):
    """
    Use MoGe to obtain intrinsics and a reference depth distribution, then replace depth with DAD.
    Always return a full-white mask.
    Returns:
        - image_b1chw_float in [-1, 1]
        - depth_b11hw (float, clamped)
        - mask_b11hw (all True)
        - initial_w2c_b144 (eye)
        - intrinsics_b133 (pixel units, resized)
    """
    if isinstance(current_image_path_or_rgb, str):
        input_bgr = cv2.imread(current_image_path_or_rgb)
        if input_bgr is None:
            raise FileNotFoundError(f"Input image not found: {current_image_path_or_rgb}")
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
    else:
        input_rgb = current_image_path_or_rgb
    del current_image_path_or_rgb
    depth_pred_h, depth_pred_w = 720, 1280
    img_resized = cv2.resize(input_rgb, (depth_pred_w, depth_pred_h))
    img_chw_0_1 = torch.tensor(img_resized / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with _timed("MoGe infer (initial)"):
        moge_out = moge_model.infer(img_chw_0_1)
    ref_depth_hw_full = moge_out["depth"]
    intrinsics_33_full_norm = moge_out["intrinsics"]
    intr_pix = intrinsics_33_full_norm.clone()
    intr_pix[0, 0] *= depth_pred_w
    intr_pix[1, 1] *= depth_pred_h
    intr_pix[0, 2] *= depth_pred_w
    intr_pix[1, 2] *= depth_pred_h
    height_scale = target_h / depth_pred_h
    width_scale  = target_w / depth_pred_w
    ref_depth_hw = F.interpolate(ref_depth_hw_full[None, None], size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    img_chw_target = F.interpolate(img_chw_0_1[None], size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
    image_b1chw_float = img_chw_target[None, None] * 2 - 1
    intrinsics_33 = intr_pix.clone()
    intrinsics_33[1, 1] *= height_scale
    intrinsics_33[1, 2] *= height_scale
    intrinsics_33[0, 0] *= width_scale
    intrinsics_33[0, 2] *= width_scale
    dad_depth_hw = _dad_predict_depth_hw_from_rgb_numpy(input_rgb, target_h, target_w, device, dad_processor, dad_model)
    dad_depth_hw = _rescale_depth_like_reference(dad_depth_hw, ref_depth_hw)
    depth_b11hw = dad_depth_hw[None, None, None]
    depth_b11hw = torch.nan_to_num(depth_b11hw, nan=1e4).clamp_(0, 1e4)
    H, W = int(dad_depth_hw.shape[0]), int(dad_depth_hw.shape[1])
    mask_b11hw = _full_true_mask_11hw(H, W, device)
    intr_b133 = intrinsics_33[None, None]
    w2c_b144 = torch.eye(4, dtype=torch.float32, device=device)[None, None]
    return image_b1chw_float, depth_b11hw, mask_b11hw, w2c_b144, intr_b133

def _predict_ar_depth_with_dad(
    image_tensor_chw_0_1: torch.Tensor,  # (C,H,W), [0,1]
    moge_model: MoGeModel,
    dad_processor: AutoImageProcessor,
    dad_model: AutoModelForDepthEstimation
):
    """
    For AR frames: compute a MoGe depth to provide reference distribution, then replace with DAD depth,
    and return an all-true mask.
    """
    with _timed("MoGe infer (AR ref)"):
        out = moge_model.infer(image_tensor_chw_0_1)
    ref_depth_hw = out["depth"]
    H, W = int(ref_depth_hw.shape[0]), int(ref_depth_hw.shape[1])
    with torch.no_grad():
        img_uint8 = (image_tensor_chw_0_1.clamp(0,1).permute(1,2,0) * 255.0).round().to(torch.uint8).cpu().numpy()
    dad_depth_hw = _dad_predict_depth_hw_from_rgb_numpy(img_uint8, H, W, image_tensor_chw_0_1.device, dad_processor, dad_model)
    dad_depth_hw = _rescale_depth_like_reference(dad_depth_hw, ref_depth_hw)
    depth_11hw = dad_depth_hw[None, None]
    depth_11hw = torch.nan_to_num(depth_11hw, nan=1e4).clamp_(0, 1e4)
    mask_11hw = _full_true_mask_11hw(H, W, image_tensor_chw_0_1.device)
    return depth_11hw, mask_11hw

# ----------------------------
# Main
# ----------------------------
def demo(args):
    """
    End-to-end:
      1) MoGe predicts intrinsics (and reference depth stats); depth is replaced with DAD; mask is full white.
      2) Cache3D buffers the initial frame for geometric warping along a camera trajectory.
      3) Render warped conditioning frames (the "input video" to diffusion).
      4) Gen3C synthesizes frames chunk-by-chunk; AR passes update Cache3D with latest frame's DAD depth.
      5) Optionally save depth (PNG16), mask (all white PNG), conditioning MP4, and final MP4.
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
    with _timed("Load MoGe"):
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    with _timed(f"Load DAD model"):
        dad_processor = AutoImageProcessor.from_pretrained(args.dad_model_id)
        dad_model = AutoModelForDepthEstimation.from_pretrained(args.dad_model_id).to(device).eval()
    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        prompts = [{"prompt": args.prompt or "", "visual_input": args.input_image_path}]
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
        log.info(f"Item {i}: input={current_image_path}, movement={args.movement_distance}, frames={args.num_video_frames}, chunk={sample_n_frames}, guidance={args.guidance}")
        with _timed("Depth+intrinsics init (MoGe+DAD)"):
            (image_b1chw_float,
             depth_b11hw,
             mask_b11hw,  # all white
             initial_w2c_b144,
             intrinsics_b133) = _predict_initial_image_depth_intrinsics_with_dad(
                current_image_path, args.height, args.width, device, moge_model, dad_processor, dad_model
            )
        if args.save_depth_dir:
            _save_depth_png16(depth_b11hw[0,0,0], _full_true_mask_hw(args.height, args.width, device), os.path.join(args.save_depth_dir, f"{args.video_save_name}_depth_{0:04d}.png"))
        if args.save_mask_dir:
            _save_mask_png(_full_true_mask_hw(args.height, args.width, device), os.path.join(args.save_mask_dir, f"{args.video_save_name}_mask_{0:04d}.png"))
        with _timed("Cache3D init"):
            cache = Cache3D_Buffer(
                frame_buffer_max=frame_buffer_max,
                generator=generator,
                noise_aug_strength=args.noise_aug_strength,
                input_image=image_b1chw_float[:, 0].clone(),
                input_depth=depth_b11hw[:, 0],
                # input_mask=mask_b11hw[:, 0],  # keep disabled
                input_w2c=initial_w2c_b144[:, 0],
                input_intrinsics=intrinsics_b133[:, 0],
                filter_points_threshold=args.filter_points_threshold,
                foreground_masking=args.foreground_masking,
            )
        initial_cam_w2c_for_traj = initial_w2c_b144[0, 0]
        initial_cam_intrinsics_for_traj = intrinsics_b133[0, 0]
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
        with _timed(f"Render cache [0:{sample_n_frames}]"):
            rendered_warp_images, rendered_warp_masks = cache.render_cache(
                generated_w2cs[:, 0:sample_n_frames],
                generated_intrinsics[:, 0:sample_n_frames],
            )
        conditioning_chunks = []
        if args.save_conditioning_video:
            conditioning_chunks.append(rendered_warp_images.clone().cpu())
        all_rendered_warps_for_sbs = []
        if args.save_buffer:
            all_rendered_warps_for_sbs.append(rendered_warp_images.clone().cpu())
        with _timed(f"Pipeline.generate [0:{sample_n_frames}]"):
            generated_output = pipeline.generate(
                prompt=current_prompt,
                image_path=current_image_path,
                negative_prompt=args.negative_prompt,
                rendered_warp_images=rendered_warp_images,
                rendered_warp_masks=rendered_warp_masks,
            )
        if generated_output is None:
            log.critical("Guardrail blocked generation.")
            continue
        video, prompt_text = generated_output
        num_ar_iterations = (generated_w2cs.shape[1] - 1) // (sample_n_frames - 1)
        for num_iter in range(1, num_ar_iterations):
            start_frame_idx = num_iter * (sample_n_frames - 1)
            end_frame_idx = start_frame_idx + sample_n_frames
            log.info(f"AR pass {num_iter}/{num_ar_iterations - 1}: frames [{start_frame_idx}:{end_frame_idx}]")
            last_frame_hwc_0_255 = torch.tensor(video[-1], device=device)
            frame_chw_0_1 = last_frame_hwc_0_255.permute(2, 0, 1).float() / 255.0
            with _timed("DAD depth predict (AR, MoGe-ref)"):
                pred_depth_11, pred_mask_11 = _predict_ar_depth_with_dad(frame_chw_0_1, moge_model, dad_processor, dad_model)
            if args.save_depth_dir:
                _save_depth_png16(pred_depth_11[0, 0], _full_true_mask_hw(pred_depth_11.shape[-2], pred_depth_11.shape[-1], device), os.path.join(args.save_depth_dir, f"{args.video_save_name}_depth_{start_frame_idx:04d}.png"))
            if args.save_mask_dir:
                _save_mask_png(_full_true_mask_hw(pred_depth_11.shape[-2], pred_depth_11.shape[-1], device), os.path.join(args.save_mask_dir, f"{args.video_save_name}_mask_{start_frame_idx:04d}.png"))
            with _timed("Cache3D update"):
                cache.update_cache(
                    new_image=frame_chw_0_1[None] * 2 - 1,
                    new_depth=pred_depth_11,
                    # new_mask=pred_mask_11,  # keep disabled
                    new_w2c=generated_w2cs[:, start_frame_idx],
                    new_intrinsics=generated_intrinsics[:, start_frame_idx],
                )
            current_w2cs = generated_w2cs[:, start_frame_idx:end_frame_idx]
            current_intr = generated_intrinsics[:, start_frame_idx:end_frame_idx]
            with _timed(f"Render cache [{start_frame_idx}:{end_frame_idx}]"):
                rendered_warp_images, rendered_warp_masks = cache.render_cache(current_w2cs, current_intr)
            if args.save_conditioning_video:
                conditioning_chunks.append(rendered_warp_images[:, 1:].clone().cpu())
            if args.save_buffer:
                all_rendered_warps_for_sbs.append(rendered_warp_images[:, 1:].clone().cpu())
            frame_bcthw_minus1_1 = frame_chw_0_1[None, None] * 2 - 1
            with _timed(f"Pipeline.generate [{start_frame_idx}:{end_frame_idx}]"):
                generated_output = pipeline.generate(
                    prompt=current_prompt,
                    image_path=frame_bcthw_minus1_1,
                    negative_prompt=args.negative_prompt,
                    rendered_warp_images=rendered_warp_images,
                    rendered_warp_masks=rendered_warp_masks,
                )
            video_new, prompt_text = generated_output
            video = np.concatenate([video, video_new[1:]], axis=0)
        final_video_to_save = video
        final_width = args.width
        if args.save_buffer and all_rendered_warps_for_sbs:
            squeezed = [t.squeeze(0) for t in all_rendered_warps_for_sbs]
            n_max = max(t.shape[0] for t in squeezed)
            padded_list = []
            for t in squeezed:
                pad_n = n_max - t.shape[0]
                pad_spec = (0, 0, 0, 0, 0, 0, 0, pad_n)
                t_CHNW = t.permute(1, 2, 0, 3)
                t_pad = F.pad(t_CHNW, pad_spec, mode='constant', value=-1.0)
                padded_list.append(t_pad)
            cat_CHNW = torch.cat(padded_list, dim=0)
            cat_NCHW = cat_CHNW.permute(2, 0, 1, 3)
            cat_NCHW = ((cat_NCHW * 0.5 + 0.5) * 255.0).clamp(0, 255).byte()
            cat_NHWC = cat_NCHW.permute(0, 2, 3, 1).cpu().numpy()
            final_video_to_save = np.concatenate([cat_NHWC, final_video_to_save], axis=2)
            final_width = final_video_to_save.shape[2]
            log.info(f"Concatenated with warp buffers. Final width = {final_width}")
        video_save_path = os.path.join(args.video_save_folder, f"{i if args.batch_input_path else args.video_save_name}.mp4")
        with _timed("Save video"):
            save_video(video=final_video_to_save, fps=args.fps, H=args.height, W=final_width, video_save_quality=5, video_save_path=video_save_path)
        log.info(f"Saved video to {video_save_path}")
        if args.save_conditioning_video and conditioning_chunks:
            try:
                for idx, t in enumerate(conditioning_chunks):
                    log.info(f"[COND] chunk[{idx}] dtype={t.dtype} device={t.device} shape={tuple(t.shape)}")
                cond = torch.cat(conditioning_chunks, dim=1)
                if cond.dim() < 4:
                    raise RuntimeError(f"Conditioning tensor has too few dims: {cond.dim()} (shape={tuple(cond.shape)})")
                C, H, W = int(cond.shape[-3]), int(cond.shape[-2]), int(cond.shape[-1])
                N = int(np.prod(cond.shape[:-3]))
                cond_NCHW = cond.reshape(N, C, H, W)
                log.info(f"[COND] after reshape to (N,C,H,W): {tuple(cond_NCHW.shape)}")
                cond_THWC = ((cond_NCHW.permute(0, 2, 3, 1).float() * 0.5 + 0.5) * 255.0).clamp(0, 255).byte().cpu().numpy()
                log.info(f"[COND] final THWC array shape: {cond_THWC.shape}")
                cond_path = os.path.join(args.video_save_folder, args.conditioning_video_name)
                with _timed("Save conditioning video"):
                    save_video(video=cond_THWC, fps=args.fps, H=args.height, W=args.width, video_save_quality=5, video_save_path=cond_path)
                log.info(f"Saved conditioning video to {cond_path}")
            except Exception as e:
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
