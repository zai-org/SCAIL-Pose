import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import glob
import shutil
import time
import traceback

import cv2
import numpy as np

try:
    import moviepy.editor as mpy
except Exception:
    import moviepy as mpy


def find_ref_image(subdir):
    for name in ('ref_image.jpg', 'ref_image.png', 'ref.jpg', 'ref.png'):
        p = os.path.join(subdir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No reference image found in {subdir}")


def save_colored_mask_image(masks, colors, out_path, bg_color=(0, 0, 0)):
    """Black bg by default; mask regions filled with solid color."""
    H, W = masks[0].shape[1:]
    frame = np.full((H, W, 3), bg_color, dtype=np.uint8)
    for mask_t, color in zip(masks, colors):
        frame[mask_t[0]] = color
    cv2.imwrite(out_path, frame)


def save_real_pixel_mask_image(masks, frame_rgb, out_path):
    """Black bg; mask regions show real pixels from frame_rgb (H,W,3 RGB)."""
    H, W = masks[0].shape[1:]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    frame_bgr = frame_rgb[:, :, ::-1]
    for mask_t in masks:
        canvas[mask_t[0]] = frame_bgr[mask_t[0]]
    cv2.imwrite(out_path, canvas)


def write_colored_mask_video(masks, colors, out_path, fps, bg_color=(255, 255, 255)):
    T = masks[0].shape[0]
    H, W = masks[0].shape[1:]
    rgb_colors = [(int(c[2]), int(c[1]), int(c[0])) for c in colors]
    bg_rgb = (int(bg_color[2]), int(bg_color[1]), int(bg_color[0]))

    frames = []
    for t in range(T):
        frame = np.full((H, W, 3), bg_rgb, dtype=np.uint8)
        for mask_t, rgb in zip(masks, rgb_colors):
            frame[mask_t[t]] = rgb
        frames.append(frame)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mpy.ImageSequenceClip(frames, fps=fps).write_videofile(out_path)


def _select_closest_to_ref(drv_masks, drv_colors, ref_masks):
    """onetomany: among multiple driving tracks, pick the one whose first-frame
    mask has the highest IoU with the ref mask, after resizing ref to driving
    resolution. Returns ([selected_mask], [selected_color]).
    """
    if len(drv_masks) <= 1:
        return drv_masks, drv_colors

    H_drv, W_drv = drv_masks[0].shape[1:]
    ref_u8 = ref_masks[0][0].astype(np.uint8) * 255
    ref_resized = cv2.resize(ref_u8, (W_drv, H_drv), interpolation=cv2.INTER_NEAREST) > 127

    best_iou, best_idx = -1.0, 0
    for i, mask in enumerate(drv_masks):
        drv_first = mask[0]
        inter = int(np.logical_and(ref_resized, drv_first).sum())
        union = int(np.logical_or(ref_resized, drv_first).sum())
        iou = inter / max(union, 1)
        print(f"  onetomany IoU track {i} (color={drv_colors[i]}): {iou:.4f}")
        if iou > best_iou:
            best_iou, best_idx = iou, i

    print(f"  onetomany selected track {best_idx} (IoU={best_iou:.4f})")
    return [drv_masks[best_idx]], [drv_colors[best_idx]]


def process_one(subdir, video_name, test_mode, onetomany, predictor, image_predictor, text):
    from TrackSam3.track import get_mask_from_image, get_mask_from_video

    mp4_path = os.path.join(subdir, video_name)
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"No {video_name} found in {subdir}")

    out_path_rendered  = os.path.join(subdir, 'rendered_v2.mp4')
    out_path_mask      = os.path.join(subdir, 'replace_mask.mp4')
    ref_image_out_path = os.path.join(subdir, 'ref_image.png')
    ref_mask_path      = os.path.join(subdir, 'ref_mask.png')

    # 1) Read fps + first frame via cv2 — decord VideoReader corrupts CUDA fds before/between
    #    SAM3 calls regardless of ordering; cv2 is CUDA-agnostic and safe at any point.
    cap = cv2.VideoCapture(mp4_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_int = max(1, int(round(fps)))
    ret, first_frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read first frame from {mp4_path}")
    first_frame_rgb = first_frame_bgr[:, :, ::-1]

    # 2) Driving → full-video masks. onetomany allows 2 tracks; we pick one later via IoU.
    max_drv = 2 if onetomany else 1
    print(f"Getting driving masks from {mp4_path} (max_targets={max_drv}, text={text})...")
    drv_masks, drv_colors = get_mask_from_video(
        mp4_path, predictor, max_targets=max_drv, sort_by='x', fixed_colors=None, text=text,
    )
    if len(drv_masks) == 0:
        raise RuntimeError(f"No valid persons detected in driving {mp4_path}")
    print(f"Driving detected: {len(drv_masks)} person(s); colors={drv_colors}")

    # 3) Resolve ref_image_path: test_mode auto-generates from first driving frame
    if test_mode:
        save_real_pixel_mask_image([drv_masks[0][0:1]], first_frame_rgb, ref_image_out_path)
        print(f"[test_mode] Ref image saved (real pixels, black bg): {ref_image_out_path}")
        ref_image_path = ref_image_out_path
    else:
        ref_image_path = find_ref_image(subdir)

    # 4) Get ref masks from ref_image (same path for both modes)
    print(f"Getting ref masks from {ref_image_path}...")
    ref_masks, ref_colors = get_mask_from_image(
        ref_image_path, image_predictor, max_targets=1, sort_by='x', fixed_colors=None, text=text,
    )
    if len(ref_masks) == 0:
        raise RuntimeError(f"No qualifying person found in ref image {ref_image_path}")

    # ref_mask.png: solid-color mask on black bg
    save_colored_mask_image(ref_masks, ref_colors, ref_mask_path, bg_color=(0, 0, 0))
    print(f"  Ref mask saved: {ref_mask_path}")

    # 4.5) onetomany: pick the driving track closest to ref by IoU
    if onetomany:
        drv_masks, drv_colors = _select_closest_to_ref(drv_masks, drv_colors, ref_masks)

    # 4) rendered_v2.mp4 is always a copy of driving
    shutil.copyfile(mp4_path, out_path_rendered)
    print(f"  Copied driving → {out_path_rendered}")

    # 5) replace_mask.mp4 — white background
    print("Writing replace_mask.mp4 (white bg)...")
    write_colored_mask_video(drv_masks, drv_colors, out_path_mask, fps_int,
                             bg_color=(255, 255, 255))

    print("Done!")
    print(f"  Rendered:      {out_path_rendered}")
    print(f"  Replace mask:  {out_path_mask}")
    print(f"  Ref mask:      {ref_mask_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SCAIL replacement pipeline: SAM3 mask extraction for character replacement. '
                    'Outputs rendered_v2.mp4 (driving copy), replace_mask.mp4 (white bg), '
                    'ref_mask.png (black bg). Pass exactly one of --subdir or --input_root.'
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--subdir', type=str, default=None,
                     help='Single-example mode: path to one subdir. '
                          'Mutually exclusive with --input_root.')
    src.add_argument('--input_root', type=str, default=None,
                     help='Batch mode: directory whose immediate subdirs are each an example. '
                          'Mutually exclusive with --subdir.')
    parser.add_argument('--video_name', type=str, default='driving.mp4',
                        choices=['driving.mp4', 'GT.mp4'],
                        help='Filename of the driving video inside each subdir.')
    parser.add_argument('--test_mode', action='store_true',
                        help='Use driving first frame as ref: saves ref_image.png with real '
                             'pixels inside mask area (black outside). No ref_image file needed.')
    parser.add_argument('--onetomany', action='store_true',
                        help='Driving may contain 2 persons; ref has 1. Picks the driving '
                             'track whose first-frame mask has highest IoU with the ref mask '
                             '(after resizing ref to driving resolution). Other tracks are dropped.')
    parser.add_argument('--text', type=str, nargs='+',
                        default=['human', 'character'],
                        help='Text prompts passed to SAM3 for both driving and ref. Add extras '
                             'like "bear" if the subject is a non-human character.')
    parser.add_argument('--skip_existing', action='store_true',
                        help='In --input_root mode, skip subdirs whose replace_mask.mp4 already exists.')
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
        if args.skip_existing and os.path.exists(os.path.join(subdir, 'replace_mask.mp4')):
            print(f"[{i+1}/{len(subdirs)}] skip (already done): {subdir}")
            n_skip += 1
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(subdirs)}] {subdir}  (video_name={args.video_name}, test_mode={args.test_mode}, onetomany={args.onetomany})")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            process_one(subdir, args.video_name, args.test_mode, args.onetomany,
                        predictor, image_predictor, args.text)
            n_ok += 1
            print(f"  -> ok ({time.time() - t0:.1f}s)")
        except Exception as e:
            n_err += 1
            print(f"  -> FAILED: {e}")
            traceback.print_exc()

    print(f"\nDone. ok={n_ok}  skipped={n_skip}  failed={n_err}  total={len(subdirs)}")
