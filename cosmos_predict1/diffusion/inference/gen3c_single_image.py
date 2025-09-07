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
import torch
import numpy as np
from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments, check_input_frames
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
import torch.nn.functional as F
import time
from contextlib import contextmanager
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import math

torch.enable_grad(False)

NUM_FRAMES = 121

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
    mask_u8 = (mask_hw_bool.to(torch.uint8) * 255).cpu().numpy()
    cv2.imwrite(out_path, mask_u8)

def _save_depth_png16(depth_hw: torch.Tensor, valid_mask_hw: torch.Tensor | None, out_path: str, q_lo: float = 0.01, q_hi: float = 0.99):
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
    return torch.ones((1,1,H,W), dtype=torch.bool, device=device)

def _full_true_mask_hw(H: int, W: int, device: torch.device) -> torch.Tensor:
    return torch.ones((H,W), dtype=torch.bool, device=device)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image-to-video demo (MoGe-free, fixed volume, 121 frames hardcoded)")
    add_common_arguments(parser)
    parser.add_argument("--prompt_upsampler_dir", type=str, default="Pixtral-12B", help="Prompt upsampler weights directory relative to checkpoint_dir")
    parser.add_argument("--input_image_path", type=str, help="Input image path for generating a single video")
    parser.add_argument("--trajectory", type=str, choices=["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise", "none"], default="left", help="Camera path for synthetic motion.")
    parser.add_argument("--camera_rotation", type=str, choices=["center_facing", "no_rotation", "trajectory_aligned"], default="center_facing", help="Rotation behavior during movement.")
    parser.add_argument("--movement_distance", type=float, default=0.3, help="Distance of the camera from the scene center (world units).")
    parser.add_argument("--noise_aug_strength", type=float, default=0.0, help="Noise augmentation on warped frames.")
    parser.add_argument("--save_buffer", action="store_true", help="If set, append rendered warp buffers beside output video.")
    parser.add_argument("--filter_points_threshold", type=float, default=0.05, help="Filter threshold for point continuity in warps.")
    parser.add_argument("--foreground_masking", action="store_true", help="Use foreground masking for warps.")
    parser.add_argument("--save_depth_dir", type=str, default=None, help="If set, saves depth PNG16 frames here (e.g., outputs/depth).")
    parser.add_argument("--save_mask_dir", type=str, default=None, help="If set, saves mask PNG frames here (e.g., outputs/mask).")
    parser.add_argument("--save_conditioning_video", action="store_true", help="If set, saves the camera-warped conditioning sequence as MP4.")
    parser.add_argument("--conditioning_video_name", type=str, default="input_conditioning.mp4", help="Filename for the conditioning MP4 (in video_save_folder).")
    parser.add_argument("--dad_model_id", type=str, default="xingyang1/Distill-Any-Depth-Large-hf", help="Hugging Face model id for Distill-Any-Depth.")
    parser.add_argument("--fixed_depth_min", type=float, default=0.3, help="Canonical near-plane depth for mapping DAD depth (world units).")
    parser.add_argument("--fixed_depth_max", type=float, default=3.0, help="Canonical far-plane depth for mapping DAD depth (world units).")
    parser.add_argument("--rescale_percentiles_lo", type=float, default=0.05, help="Low percentile of DAD used for robust scaling.")
    parser.add_argument("--rescale_percentiles_hi", type=float, default=0.95, help="High percentile of DAD used for robust scaling.")
    parser.add_argument("--clamp_final_depth", action="store_true", help="Clamp final depth to [fixed_depth_min,fixed_depth_max] to enforce consistent particle volume.")
    parser.add_argument("--intrinsics_mode", type=str, choices=["fov", "explicit"], default="fov", help="How to set intrinsics without MoGe.")
    parser.add_argument("--fov_deg", type=float, default=60.0, help="Horizontal field-of-view in degrees when intrinsics_mode=fov.")
    parser.add_argument("--fx", type=float, default=None, help="Explicit fx in pixels when intrinsics_mode=explicit.")
    parser.add_argument("--fy", type=float, default=None, help="Explicit fy in pixels when intrinsics_mode=explicit.")
    parser.add_argument("--cx", type=float, default=None, help="Explicit cx in pixels when intrinsics_mode=explicit.")
    parser.add_argument("--cy", type=float, default=None, help="Explicit cy in pixels when intrinsics_mode=explicit.")
    return parser

def parse_arguments() -> argparse.Namespace:
    parser = create_parser()
    return parser.parse_args()

def validate_args(args):
    assert args.fixed_depth_min < args.fixed_depth_max, "fixed_depth_min must be < fixed_depth_max"
    assert 0.0 <= args.rescale_percentiles_lo < args.rescale_percentiles_hi <= 1.0, "rescale_percentiles_lo < rescale_percentiles_hi and both in [0,1]"
    if args.intrinsics_mode == "fov":
        assert 1.0 < args.fov_deg < 179.0, "fov_deg must be in (1,179)"
    else:
        for name in ("fx","fy","cx","cy"):
            assert getattr(args, name) is not None, f"{name} must be provided when intrinsics_mode=explicit"
        assert args.fx > 0 and args.fy > 0, "fx and fy must be > 0"

def _dad_predict_depth_hw_from_rgb_numpy(input_image_rgb: np.ndarray, target_h: int, target_w: int, device: torch.device, dad_processor: AutoImageProcessor, dad_model: AutoModelForDepthEstimation) -> torch.Tensor:
    with _timed("DAD infer"):
        inputs = dad_processor(images=[input_image_rgb], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = dad_model(**inputs)
        post = dad_processor.post_process_depth_estimation(outputs, target_sizes=[(target_h, target_w)])
        depth_hw = post[0]["predicted_depth"].to(device).to(torch.float32)
    return 1.0 - depth_hw

def _rescale_depth_fixed(dad_hw: torch.Tensor, fixed_lo: float, fixed_hi: float, q_lo: float, q_hi: float, clamp_final: bool, tag: str) -> torch.Tensor:
    device = dad_hw.device
    eps = torch.tensor(1e-6, device=device, dtype=torch.float32)
    q_dad = torch.quantile(dad_hw.reshape(-1), torch.tensor([q_lo, q_hi], device=device))
    den = (q_dad[1] - q_dad[0]).clamp_min(eps)
    dad_norm = (dad_hw - q_dad[0]) / den
    target_lo_t = torch.tensor(float(fixed_lo), device=device, dtype=torch.float32)
    target_hi_t = torch.tensor(float(fixed_hi), device=device, dtype=torch.float32)
    scaled = dad_norm * (target_hi_t - target_lo_t) + target_lo_t
    if clamp_final:
        scaled = scaled.clamp(min=float(fixed_lo), max=float(fixed_hi))
    scaled = torch.nan_to_num(scaled, nan=1e4).clamp_(0.0, 1e4)
    try:
        alpha = (target_hi_t - target_lo_t) / den
        log.info(f"[DEPTH SCALE][{tag}] fixed_volume=({float(target_lo_t):.6f},{float(target_hi_t):.6f}) dad_p=({float(q_dad[0]):.6f},{float(q_dad[1]):.6f}) alpha={float(alpha):.6f} clamp={clamp_final}")
    except Exception:
        pass
    return scaled

def _make_intrinsics(target_h: int, target_w: int, mode: str, fov_deg: float | None, fx: float | None, fy: float | None, cx: float | None, cy: float | None, device: torch.device) -> torch.Tensor:
    if mode == "fov":
        fov_rad = math.radians(float(fov_deg))
        fx_px = 0.5 * float(target_w) / max(math.tan(0.5 * fov_rad), 1e-6)
        fy_px = fx_px
        cx_px = (float(target_w) - 1.0) * 0.5
        cy_px = (float(target_h) - 1.0) * 0.5
    else:
        fx_px = float(fx)
        fy_px = float(fy)
        cx_px = float(cx)
        cy_px = float(cy)
    K = torch.tensor([[fx_px, 0.0, cx_px], [0.0, fy_px, cy_px], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    log.info(f"[INTRINSICS] mode={mode} fx={fx_px:.3f} fy={fy_px:.3f} cx={cx_px:.3f} cy={cy_px:.3f}")
    return K

def _predict_initial_image_depth_intrinsics_with_dad_only(current_image_path_or_rgb: str | np.ndarray, target_h: int, target_w: int, device: torch.device, dad_processor: AutoImageProcessor, dad_model: AutoModelForDepthEstimation, fixed_depth_min: float, fixed_depth_max: float, rescale_percentiles_lo: float, rescale_percentiles_hi: float, clamp_final_depth: bool, intrinsics_mode: str, fov_deg: float | None, fx: float | None, fy: float | None, cx: float | None, cy: float | None):
    if isinstance(current_image_path_or_rgb, str):
        input_bgr = cv2.imread(current_image_path_or_rgb)
        if input_bgr is None:
            raise FileNotFoundError(f"Input image not found: {current_image_path_or_rgb}")
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
    else:
        input_rgb = current_image_path_or_rgb
    del current_image_path_or_rgb
    img_resized = cv2.resize(input_rgb, (target_w, target_h))
    img_chw_0_1 = torch.tensor(img_resized / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    image_b1chw_float = img_chw_0_1[None, None] * 2 - 1
    dad_depth_hw = _dad_predict_depth_hw_from_rgb_numpy(img_resized, target_h, target_w, device, dad_processor, dad_model)
    dad_depth_hw = _rescale_depth_fixed(dad_depth_hw, fixed_depth_min, fixed_depth_max, rescale_percentiles_lo, rescale_percentiles_hi, clamp_final_depth, tag="init")
    depth_b11hw = torch.nan_to_num(dad_depth_hw[None, None, None], nan=1e4).clamp_(0, 1e4)
    mask_b11hw = _full_true_mask_11hw(target_h, target_w, device)
    intrinsics_33 = _make_intrinsics(target_h, target_w, intrinsics_mode, fov_deg, fx, fy, cx, cy, device)
    intr_b133 = intrinsics_33[None, None]
    w2c_b144 = torch.eye(4, dtype=torch.float32, device=device)[None, None]
    return image_b1chw_float, depth_b11hw, mask_b11hw, w2c_b144, intr_b133

def demo(args):
    """
    End-to-end (MoGe-free, 121 frames):
      1) Intrinsics are computed from FOV or explicit fx/fy/cx/cy; depth is DAD mapped to a fixed volume [near,far]; mask is full white.
      2) Cache3D buffers the initial frame and renders a single 121-frame trajectory without AR passes.
      3) Gen3C synthesizes exactly 121 frames conditioned on the rendered warps.
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
        pipeline = Gen3cPipeline(inference_type="video2world", checkpoint_dir=args.checkpoint_dir, checkpoint_name="Gen3C-Cosmos-7B", prompt_upsampler_dir=args.prompt_upsampler_dir, enable_prompt_upsampler=not args.disable_prompt_upsampler, offload_network=args.offload_diffusion_transformer, offload_tokenizer=args.offload_tokenizer, offload_text_encoder_model=args.offload_text_encoder_model, offload_prompt_upsampler=args.offload_prompt_upsampler, offload_guardrail_models=args.offload_guardrail_models, disable_guardrail=args.disable_guardrail, disable_prompt_encoder=args.disable_prompt_encoder, guidance=args.guidance, num_steps=args.num_steps, height=args.height, width=args.width, fps=args.fps, num_video_frames=NUM_FRAMES, seed=args.seed)
    frame_buffer_max = pipeline.model.frame_buffer_max
    sample_n_frames = pipeline.model.chunk_size
    assert sample_n_frames == NUM_FRAMES, f"Pipeline chunk_size={sample_n_frames} but this script is hardcoded for {NUM_FRAMES}."
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with _timed("Load DAD model"):
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
        log.info(f"Item {i}: input={current_image_path}, movement={args.movement_distance}, frames={NUM_FRAMES}, chunk={sample_n_frames}, guidance={args.guidance}")
        with _timed("Depth+intrinsics init (DAD only, fixed volume)"):
            image_b1chw_float, depth_b11hw, mask_b11hw, initial_w2c_b144, intrinsics_b133 = _predict_initial_image_depth_intrinsics_with_dad_only(current_image_path, args.height, args.width, device, dad_processor, dad_model, args.fixed_depth_min, args.fixed_depth_max, args.rescale_percentiles_lo, args.rescale_percentiles_hi, args.clamp_final_depth, args.intrinsics_mode, args.fov_deg, args.fx, args.fy, args.cx, args.cy)
        if args.save_depth_dir:
            _save_depth_png16(depth_b11hw[0,0,0], _full_true_mask_hw(args.height, args.width, device), os.path.join(args.save_depth_dir, f"{args.video_save_name}_depth_{0:04d}.png"))
        if args.save_mask_dir:
            _save_mask_png(_full_true_mask_hw(args.height, args.width, device), os.path.join(args.save_mask_dir, f"{args.video_save_name}_mask_{0:04d}.png"))
        with _timed("Cache3D init"):
            cache = Cache3D_Buffer(frame_buffer_max=frame_buffer_max, generator=generator, noise_aug_strength=args.noise_aug_strength, input_image=image_b1chw_float[:, 0].clone(), input_depth=depth_b11hw[:, 0], input_w2c=initial_w2c_b144[:, 0], input_intrinsics=intrinsics_b133[:, 0], filter_points_threshold=args.filter_points_threshold, foreground_masking=args.foreground_masking)
        initial_cam_w2c_for_traj = initial_w2c_b144[0, 0]
        initial_cam_intrinsics_for_traj = intrinsics_b133[0, 0]
        center_depth = float(0.5 * (args.fixed_depth_min + args.fixed_depth_max))
        with _timed("Trajectory generation"):
            generated_w2cs, generated_intrinsics = generate_camera_trajectory(trajectory_type=args.trajectory, initial_w2c=initial_cam_w2c_for_traj, initial_intrinsics=initial_cam_intrinsics_for_traj, num_frames=NUM_FRAMES, movement_distance=args.movement_distance, camera_rotation=args.camera_rotation, center_depth=center_depth, device=device.type)
        with _timed("Render cache [0:121]"):
            rendered_warp_images, rendered_warp_masks = cache.render_cache(generated_w2cs[:, 0:sample_n_frames], generated_intrinsics[:, 0:sample_n_frames])
        if args.save_conditioning_video:
            try:
                cond = rendered_warp_images.clone().cpu()
                C, H, W = int(cond.shape[-3]), int(cond.shape[-2]), int(cond.shape[-1])
                N = int(np.prod(cond.shape[:-3]))
                cond_NCHW = cond.reshape(N, C, H, W)
                cond_THWC = ((cond_NCHW.permute(0, 2, 3, 1).float() * 0.5 + 0.5) * 255.0).clamp(0, 255).byte().cpu().numpy()
                cond_path = os.path.join(args.video_save_folder, args.conditioning_video_name)
                with _timed("Save conditioning video"):
                    save_video(video=cond_THWC, fps=args.fps, H=args.height, W=args.width, video_save_quality=5, video_save_path=cond_path)
                log.info(f"Saved conditioning video to {cond_path}")
            except Exception as e:
                log.exception(f"[COND] Failed to save conditioning video; continuing without it. Error: {e}")
        with _timed("Pipeline.generate [0:121]"):
            generated_output = pipeline.generate(prompt=current_prompt, image_path=current_image_path, negative_prompt=args.negative_prompt, rendered_warp_images=rendered_warp_images, rendered_warp_masks=rendered_warp_masks)
        if generated_output is None:
            log.critical("Guardrail blocked generation.")
            continue
        video, prompt_text = generated_output
        final_video_to_save = video
        final_width = args.width
        if args.save_buffer:
            try:
                t = rendered_warp_images.clone().cpu()
                t = t.squeeze(0)
                t_CHNW = t.permute(1, 2, 0, 3)
                t_pad = t_CHNW
                cat_CHNW = t_pad
                cat_NCHW = cat_CHNW.permute(2, 0, 1, 3)
                cat_NCHW = ((cat_NCHW * 0.5 + 0.5) * 255.0).clamp(0, 255).byte()
                cat_NHWC = cat_NCHW.permute(0, 2, 3, 1).cpu().numpy()
                final_video_to_save = np.concatenate([cat_NHWC, final_video_to_save], axis=2)
                final_width = final_video_to_save.shape[2]
                log.info(f"Concatenated with warp buffers. Final width = {final_width}")
            except Exception as e:
                log.exception(f"[SBS] Failed to append warp buffers; continuing without them. Error: {e}")
        video_save_path = os.path.join(args.video_save_folder, f"{i if args.batch_input_path else args.video_save_name}.mp4")
        with _timed("Save video"):
            save_video(video=final_video_to_save, fps=args.fps, H=args.height, W=final_width, video_save_quality=5, video_save_path=video_save_path)
        log.info(f"Saved video to {video_save_path}")
        log.info(f"Item {i} total elapsed: {(time.perf_counter() - t_item_start):.3f}s")
    log.info(f"Overall elapsed: {(time.perf_counter() - t_total_start):.3f}s")

if __name__ == "__main__":
    args = parse_arguments()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    demo(args)
