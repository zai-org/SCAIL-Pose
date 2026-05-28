import os
import sys

# 动态添加项目根目录到 sys.path，这样就不需要 export PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # SCAIL_Pose 目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import glob
import shutil
import time
import traceback

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu

try:
    import moviepy.editor as mpy
except Exception:
    import moviepy as mpy


def find_ref_image(subdir):
    for name in ('ref_image.jpg', 'ref_image.png', 'ref.jpg', 'ref.png'):
        p = os.path.join(subdir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No reference image (ref_image.jpg/png or ref.jpg) found in {subdir}")


def save_colored_mask_image(masks, colors, out_path):
    """Render the first frame of each person's mask onto a single colored image (BGR)."""
    H, W = masks[0].shape[1:]
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    for mask_t, color in zip(masks, colors):
        frame[mask_t[0]] = color
    cv2.imwrite(out_path, frame)


def write_colored_mask_video(masks, colors, out_path, fps):
    """Write a per-frame colored mask mp4 (each person painted in their BGR color)."""
    T = masks[0].shape[0]
    H, W = masks[0].shape[1:]
    rgb_colors = [(int(c[2]), int(c[1]), int(c[0])) for c in colors]  # BGR -> RGB for moviepy

    frames = []
    for t in range(T):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        for mask_t, rgb in zip(masks, rgb_colors):
            frame[mask_t[t]] = rgb
        frames.append(frame)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mpy.ImageSequenceClip(frames, fps=fps).write_videofile(out_path)


def process_one(subdir, video_name, e2e_mode, max_persons,
                model_nlf, detector, predictor, image_predictor):
    from TrackSam3.track import get_mask_from_image, get_mask_from_video

    mp4_path = os.path.join(subdir, video_name)
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"No {video_name} found in {subdir}")

    ref_image_path = find_ref_image(subdir)

    out_path_rendered = os.path.join(subdir, 'rendered_v2.mp4')
    out_path_mask     = os.path.join(subdir, 'rendered_mask_v2.mp4')
    ref_mask_path     = os.path.join(subdir, 'ref_mask.jpg')

    # 1) Driving is the canonical authority for person count (video tracker is reliable;
    #    SAM3 image mode tends to return duplicate masks for the same character).
    print(f"Getting driving masks from {mp4_path}...")
    drv_masks, drv_colors = get_mask_from_video(
        mp4_path, predictor, max_targets=max_persons, sort_by='x', fixed_colors=None,
    )
    if len(drv_masks) == 0:
        raise RuntimeError(f"No valid persons detected in driving {mp4_path}")
    N = len(drv_masks)
    print(f"Driving defines {N} person(s) (capped at --max_persons={max_persons}); colors={drv_colors}")

    # 2) Ref provides per-person colors; cap at N and sort left-to-right to match driving.
    print(f"Getting ref masks from {ref_image_path}...")
    ref_masks, ref_colors = get_mask_from_image(
        ref_image_path, image_predictor, max_targets=N, sort_by='x', fixed_colors=None,
    )
    if len(ref_masks) < N:
        raise RuntimeError(
            f"Ref has only {len(ref_masks)} qualifying person(s) but driving has {N}. "
            f"Cannot color-align."
        )
    save_colored_mask_image(ref_masks, ref_colors, ref_mask_path)
    print(f"  Ref mask saved: {ref_mask_path}")

    # 3) Read driving frames and fps once
    vr = VideoReader(mp4_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    fps_int = max(1, int(round(fps)))
    video_frames_np = vr.get_batch(list(range(len(vr)))).asnumpy()  # (T, H, W, 3) RGB

    # 4) Branch on e2e_mode
    if e2e_mode:
        print("[e2e_mode] copying driving as rendered_v2.mp4 ...")
        shutil.copyfile(mp4_path, out_path_rendered)
        print("[e2e_mode] writing colored mask video as rendered_mask_v2.mp4 ...")
        write_colored_mask_video(drv_masks, drv_colors, out_path_mask, fps_int)
    else:
        from NLFPoseExtract.nlf_render import run_nlf_from_masks
        print("Running NLF and rendering skeletons ...")
        run_nlf_from_masks(
            video_frames=video_frames_np,
            masks=drv_masks,
            colors=drv_colors,
            model_nlf=model_nlf,
            nlf_render_path=out_path_rendered,
            nlf_render_mask_path=out_path_mask,
            fps=fps_int,
            detector=detector,
        )

    print("Done!")
    print(f"  Rendered:      {out_path_rendered}")
    print(f"  Rendered mask: {out_path_mask}")
    print(f"  Ref mask:      {ref_mask_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SCAIL pose AIO: SAM3 mask + (optional) NLF skeleton render, with '
                    'ref-aligned multi-person colors. Pass exactly one of --subdir '
                    '(single example) OR --input_root (batch over a directory of examples).'
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--subdir', type=str, default=None,
                     help='Single-example mode: path to one subdir containing driving video + ref_image. '
                          'Mutually exclusive with --input_root.')
    src.add_argument('--input_root', type=str, default=None,
                     help='Batch mode: directory whose immediate subdirs are each an example. '
                          'Models load once and every subdir is processed in one invocation. '
                          'Mutually exclusive with --subdir.')
    parser.add_argument('--video_name', type=str, default='driving.mp4',
                        choices=['driving.mp4', 'GT.mp4'],
                        help='Filename of the driving video inside each subdir.')
    parser.add_argument('--e2e_mode', action='store_true',
                        help='If set, skip pose extraction: rendered_v2.mp4 is a copy of the driving video, '
                             'and rendered_mask_v2.mp4 is the colored SAM3 mask video.')
    parser.add_argument('--max_persons', type=int, default=2,
                        help='Maximum number of persons to track from ref / driving (default 2).')
    parser.add_argument('--skip_existing', action='store_true',
                        help='In --input_root mode, skip subdirs whose rendered_mask_v2.mp4 already exists.')
    parser.add_argument('--model_path', type=str,
                        default='pretrained_weights/nlf_l_multi_0.3.2.torchscript',
                        help='Path to NLF TorchScript model (only used when --e2e_mode is not set).')
    parser.add_argument('--sam3_model', type=str,
                        default='pretrained_weights/sam3.pt',
                        help='Path to SAM3 model weights.')
    args = parser.parse_args()

    from ultralytics.models.sam import SAM3SemanticPredictor, SAM3VideoSemanticPredictor

    print("Initializing SAM3 video predictor...")
    overrides = dict(
        conf=0.25, task="segment", mode="predict", imgsz=640,
        model=args.sam3_model, half=True, save=False, verbose=False,
    )
    predictor = SAM3VideoSemanticPredictor(overrides=overrides, new_det_thresh=1.0)

    print("Initializing SAM3 image predictor...")
    image_predictor = SAM3SemanticPredictor(overrides=overrides)

    if args.e2e_mode:
        model_nlf = None
        detector = None
    else:
        from DWPoseProcess.dwpose import DWposeDetector
        print("Loading NLF model...")
        model_nlf = torch.jit.load(args.model_path).cuda().eval()
        print("Loading DWpose detector...")
        detector = DWposeDetector(use_batch=False).to(0)
    print("All models loaded.")

    if args.subdir is not None:
        subdirs = [args.subdir]
    else:
        subdirs = sorted(d for d in glob.glob(os.path.join(args.input_root, '*'))
                         if os.path.isdir(d))
        if not subdirs:
            print(f"No subdirs found under {args.input_root}")
            sys.exit(0)

    n_ok, n_skip, n_err = 0, 0, 0
    for i, subdir in enumerate(subdirs):
        if args.skip_existing and os.path.exists(os.path.join(subdir, 'rendered_mask_v2.mp4')):
            print(f"[{i+1}/{len(subdirs)}] skip (already done): {subdir}")
            n_skip += 1
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(subdirs)}] {subdir}  (video_name={args.video_name}, e2e_mode={args.e2e_mode})")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            process_one(subdir, args.video_name, args.e2e_mode, args.max_persons,
                        model_nlf, detector, predictor, image_predictor)
            n_ok += 1
            print(f"  -> ok ({time.time() - t0:.1f}s)")
        except Exception as e:
            n_err += 1
            print(f"  -> FAILED: {e}")
            traceback.print_exc()

    print(f"\nDone. ok={n_ok}  skipped={n_skip}  failed={n_err}  total={len(subdirs)}")
