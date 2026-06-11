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

import numpy as np
import torch
from decord import VideoReader, cpu

from NLFPoseExtract.v2_helper import (
    find_ref_image,
    match_driving_to_ref_by_center,
    save_colored_mask_image,
    write_colored_mask_video,
    write_kept_driving_video,
)


def process_one(subdir, video_name, e2e_mode, crop_kind, max_persons, text,
                model_nlf, detector, predictor, image_predictor,
                crop_margin=0.05):
    from TrackSam3.track import get_mask_from_image_via_video, get_mask_from_video

    if crop_kind is not None and not e2e_mode:
        raise ValueError("--crop_e2e_bbox / --crop_e2e_mask / --crop_e2e_steady_bbox require --e2e_mode")

    mp4_path = os.path.join(subdir, video_name)
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"No {video_name} found in {subdir}")

    ref_image_path = find_ref_image(subdir)

    out_path_rendered = os.path.join(subdir, 'rendered_v2.mp4')
    out_path_mask     = os.path.join(subdir, 'rendered_mask_v2.mp4')
    ref_mask_path     = os.path.join(subdir, 'ref_mask.jpg')

    # 1) Driving is the canonical authority for person count (video tracker is reliable;
    #    SAM3 image mode tends to return duplicate masks for the same character).
    print(f"Getting driving masks from {mp4_path} (text={text})...")
    drv_masks, drv_colors = get_mask_from_video(
        mp4_path, predictor, max_targets=max_persons, sort_by='x', fixed_colors=None,
        text=text,
    )
    if len(drv_masks) == 0:
        raise RuntimeError(f"No valid persons detected in driving {mp4_path}")
    N = len(drv_masks)
    print(f"Driving defines {N} person(s) (capped at --max_persons={max_persons}); colors={drv_colors}")

    # 2) Ref provides per-person colors; cap at N and sort left-to-right to match driving.
    # Route ref through the video predictor (single-frame mp4 wrapper) — image-mode SAM3
    # often misses small / distant subjects that the video pipeline picks up reliably.
    print(f"Getting ref masks from {ref_image_path} (text={text})...")
    ref_masks, ref_colors = get_mask_from_image_via_video(
        ref_image_path, predictor, max_targets=N, sort_by='x', fixed_colors=None,
        text=text,
    )
    if len(ref_masks) < N:
        print(f"  Ref has {len(ref_masks)} person(s) but driving has {N}; "
              f"matching driving subset to ref by normalized center ...")
        drv_masks, drv_colors = match_driving_to_ref_by_center(
            drv_masks, drv_colors, ref_masks)
        if len(drv_masks) != len(ref_masks):
            raise RuntimeError(
                f"Center matching produced {len(drv_masks)} driving tracks but "
                f"ref has {len(ref_masks)}; cannot align."
            )
        N = len(drv_masks)
        print(f"  Reduced driving to {N} person(s).")
    save_colored_mask_image(ref_masks, ref_colors, ref_mask_path, bg_color=(255, 255, 255))
    print(f"  Ref mask saved: {ref_mask_path}")

    # 3) Read driving frames and fps once
    vr = VideoReader(mp4_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    fps_int = max(1, int(round(fps)))
    video_frames_np = vr.get_batch(list(range(len(vr)))).asnumpy()  # (T, H, W, 3) RGB

    # 4) Branch on e2e_mode (+ optional crop_kind for rendered_v2 only)
    if e2e_mode:
        if crop_kind is not None:
            print(f"[e2e_mode+crop_e2e_{crop_kind}] writing kept driving as rendered_v2.mp4 "
                  f"(bbox_margin={crop_margin}) ...")
            write_kept_driving_video(video_frames_np, drv_masks, out_path_rendered,
                                     fps_int, crop_kind=crop_kind, bbox_margin=crop_margin)
        else:
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
                        choices=['driving.mp4', 'GT.mp4', 'raw.mp4'],
                        help='Filename of the driving video inside each subdir.')
    parser.add_argument('--e2e_mode', action='store_true',
                        help='If set, skip pose extraction: rendered_v2.mp4 is a copy of the driving video, '
                             'and rendered_mask_v2.mp4 is the colored SAM3 mask video. Recommanded setting to True as it\'s more accurate and easy-to-use than pose-driven for most cases.'
                             'Default pose-driven provides more control and interpretability, you can adopt pose-driven for extremely challenging inputs.')
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument('--crop_e2e_mask', action='store_true',
                            help='Sub-flag of --e2e_mode. rendered_v2.mp4 keeps only pixels inside '
                                 'each person\'s mask silhouette; rest is black. The behaviour is between pose-driven and crop_e2e_bbox.'
                                 'For 512p e2e runs or portrait videos, as the main training is in under this resolution, typically the function is not needed.'
                                 'For 704p horizontal (especially multi-human scenarios as it\'s zero-shot), we notice the crop can be an inference optimization to reduce artifacts.'
                                 'For other cases the crop may also not be necessary, as the default full e2e is usually better especially in terms of human-object interactions.')
    crop_group.add_argument('--crop_e2e_bbox', action='store_true',
                            help='Sub-flag of --e2e_mode. rendered_v2.mp4 keeps only pixels inside '
                                 'each person\'s mask bbox (margin-padded, overlap-merged); rest is black. ')
    crop_group.add_argument('--crop_e2e_steady_bbox', action='store_true',
                            help='Sub-flag of --e2e_mode. One static bbox from the union of all person masks '
                                 'across all frames (with margin); the kept rectangle never moves. Usually not needed unless you specifically want behaviour between full e2e and crop_e2e_bbox. ')
    # All the expected behaviour of the 4 cropping alternatives (including not cropping, i.e full e2e) are based on 50 steps with cfg 4.0, when using lightx2v the results may be different
    # Anyway, the cropping is generally a harmless optimization to reduce artifacts, and as long as it can cover the main body movements and interactions with objects, it should be fine.
    parser.add_argument('--crop_margin', type=float, default=0.05,
                        help='Fractional padding added to bboxes in --crop_e2e_bbox / '
                             '--crop_e2e_steady_bbox (default 0.05).')
    parser.add_argument('--max_persons', type=int, default=2,
                        help='Maximum number of persons to track from ref / driving (default 2).')
    parser.add_argument('--text', type=str, nargs='+',
                        default=['human', 'character'],
                        help='Text prompts passed to SAM3 for both driving and ref. Add extras '
                             'like "robot arm" "gripper" if the subject is a non-human character '
                             '(e.g. a robotic arm in egocentric/animation data).')
    parser.add_argument('--skip_existing', action='store_true',
                        help='In --input_root mode, skip subdirs whose rendered_mask_v2.mp4 already exists.')
    parser.add_argument('--model_path', type=str,
                        default='pretrained_weights/nlf_l_multi_0.3.2.torchscript',
                        help='Path to NLF TorchScript model (only used when --e2e_mode is not set).')
    parser.add_argument('--sam3_model', type=str,
                        default='pretrained_weights/sam3.pt',
                        help='Path to SAM3 model weights.')
    args = parser.parse_args()

    crop_kind = ('bbox' if args.crop_e2e_bbox else
                 'mask' if args.crop_e2e_mask else
                 'steady_bbox' if args.crop_e2e_steady_bbox else None)
    if crop_kind is not None and not args.e2e_mode:
        parser.error("--crop_e2e_bbox / --crop_e2e_mask / --crop_e2e_steady_bbox require --e2e_mode")

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
        print(f"[{i+1}/{len(subdirs)}] {subdir}  (video_name={args.video_name}, e2e_mode={args.e2e_mode}, crop_kind={crop_kind})")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            process_one(subdir, args.video_name, args.e2e_mode, crop_kind,
                        args.max_persons, args.text, model_nlf, detector,
                        predictor, image_predictor, crop_margin=args.crop_margin)
            n_ok += 1
            print(f"  -> ok ({time.time() - t0:.1f}s)")
        except Exception as e:
            n_err += 1
            print(f"  -> FAILED: {e}")
            traceback.print_exc()

    print(f"\nDone. ok={n_ok}  skipped={n_skip}  failed={n_err}  total={len(subdirs)}")
