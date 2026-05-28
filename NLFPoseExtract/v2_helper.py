"""Helpers shared by process_animation_aio.py and process_replacement.py.

Small, side-effect-free utilities for finding ref images, rendering SAM3 masks
(image + video), and writing the e2e "kept-region" driving video used by
--crop_e2e_bbox / --crop_e2e_mask in process_animation_aio.
"""
import os

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
    raise FileNotFoundError(f"No reference image (ref_image.jpg/png or ref.jpg) found in {subdir}")


def imread_bgr(image_path):
    """Robust BGR uint8 image reader. Tries cv2.imread first, falls back to PIL
    when cv2 returns None — handles cases where libpng's global state gets
    corrupted by SAM3/torch deps and rejects PNGs that PIL still decodes fine.

    Raises FileNotFoundError if both readers fail."""
    img = cv2.imread(str(image_path))
    if img is not None:
        return img
    try:
        from PIL import Image
        pil = Image.open(image_path).convert('RGB')
        print(f"  [warn] cv2.imread failed for {image_path}; used PIL fallback")
        return np.array(pil)[:, :, ::-1].copy()  # RGB -> BGR
    except Exception as e:
        raise FileNotFoundError(f"Cannot read image: {image_path} ({e})")


def save_colored_mask_image(masks, colors, out_path, bg_color=(0, 0, 0)):
    """Render the first frame of each person's mask onto a single BGR image.
    bg_color is BGR; default black."""
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


def write_colored_mask_video(masks, colors, out_path, fps, bg_color=(0, 0, 0)):
    """Per-frame colored mask mp4 (each person painted in their BGR color).
    bg_color is BGR; default black."""
    T = masks[0].shape[0]
    H, W = masks[0].shape[1:]
    rgb_colors = [(int(c[2]), int(c[1]), int(c[0])) for c in colors]  # BGR -> RGB for moviepy
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


def _mask_to_numpy_bool(m_t):
    if hasattr(m_t, 'cpu'):
        m_t = m_t.cpu().numpy()
    return np.asarray(m_t).astype(bool)


def match_driving_to_ref_by_center(drv_masks, drv_colors, ref_masks):
    """Greedy nearest-center matching for the one-to-many case where SAM3 finds
    more driving persons than ref (often false positives like instruments/props
    matching the 'character' prompt). For each ref person, pick the not-yet-used
    driving person whose first-frame mask centroid (normalized to [0,1]^2) is
    closest. Returns (matched_masks, matched_colors) in ref order, len == len(ref).
    """
    def norm_center(mask2d, H, W):
        mt = _mask_to_numpy_bool(mask2d)
        ys, xs = np.where(mt)
        if len(xs) == 0:
            return None
        return (float(xs.mean()) / W, float(ys.mean()) / H)

    H_d, W_d = drv_masks[0].shape[1:]
    H_r, W_r = ref_masks[0].shape[1:]
    drv_c = [norm_center(m[0], H_d, W_d) for m in drv_masks]
    ref_c = [norm_center(m[0], H_r, W_r) for m in ref_masks]

    used = set()
    out_masks, out_colors = [], []
    for ri, rc in enumerate(ref_c):
        if rc is None:
            continue
        best_i, best_d = None, float('inf')
        for di, dc in enumerate(drv_c):
            if di in used or dc is None:
                continue
            d = (dc[0] - rc[0]) ** 2 + (dc[1] - rc[1]) ** 2
            if d < best_d:
                best_d, best_i = d, di
        if best_i is None:
            continue
        used.add(best_i)
        out_masks.append(drv_masks[best_i])
        out_colors.append(drv_colors[best_i])
        print(f"  match: ref[{ri}] center={rc} -> drv[{best_i}] "
              f"center={drv_c[best_i]} (d^2={best_d:.4f})")
    return out_masks, out_colors


def _merge_overlapping_bboxes(boxes):
    """Iteratively merge any pair of overlapping bboxes into their bounding union
    until no overlaps remain. O(N^2) — fine since N <= max_persons (typically 2)."""
    def overlaps(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
    def union(a, b):
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if overlaps(boxes[i], boxes[j]):
                    boxes[i] = union(boxes[i], boxes[j])
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def _frame_keep_mask(masks_t, H, W, crop_kind, bbox_margin):
    """(H, W) bool mask of pixels to keep this frame.
    crop_kind='mask': union of per-person silhouettes.
    crop_kind='bbox': union of per-person bboxes (margin-padded, overlap-merged)."""
    if crop_kind == 'mask':
        keep = np.zeros((H, W), bool)
        for m_t in masks_t:
            keep |= _mask_to_numpy_bool(m_t)
        return keep
    boxes = []
    for m_t in masks_t:
        ys, xs = np.where(_mask_to_numpy_bool(m_t))
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        mx = int(round((x1 - x0) * bbox_margin))
        my = int(round((y1 - y0) * bbox_margin))
        boxes.append((max(0, x0 - mx), max(0, y0 - my),
                      min(W, x1 + mx), min(H, y1 + my)))
    keep = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in _merge_overlapping_bboxes(boxes):
        keep[y0:y1, x0:x1] = True
    return keep


def _compute_steady_bbox(masks, H, W, bbox_margin):
    """Single bbox covering the union of every per-person mask across all frames,
    with fractional margin. Returns (x0, y0, x1, y1) or None if all masks empty."""
    T = masks[0].shape[0]
    union = np.zeros((H, W), bool)
    for m in masks:
        for t in range(T):
            union |= _mask_to_numpy_bool(m[t])
    ys, xs = np.where(union)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    mx = int(round((x1 - x0) * bbox_margin))
    my = int(round((y1 - y0) * bbox_margin))
    return (max(0, x0 - mx), max(0, y0 - my),
            min(W, x1 + mx), min(H, y1 + my))


def write_kept_driving_video(video_frames_rgb, masks, out_path, fps,
                             crop_kind, bbox_margin=0.05):
    """Same dims as the driving video; per-frame keep only the region selected by
    crop_kind ('bbox' | 'mask' | 'steady_bbox') and blacken the rest.

    'steady_bbox' uses one static bbox = union of all masks across all frames
    (with margin), so the kept rectangle never moves — useful when the driving
    has camera motion and you want a stable crop window."""
    T = masks[0].shape[0]
    H, W = video_frames_rgb.shape[1:3]

    static_keep = None
    if crop_kind == 'steady_bbox':
        bbox = _compute_steady_bbox(masks, H, W, bbox_margin)
        static_keep = np.zeros((H, W), bool)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            static_keep[y0:y1, x0:x1] = True
            print(f"  steady_bbox: {bbox} (W={x1-x0}, H={y1-y0}) in {W}x{H} frame")

    frames = []
    for t in range(T):
        if static_keep is not None:
            keep = static_keep
        else:
            keep = _frame_keep_mask([masks[i][t] for i in range(len(masks))],
                                    H, W, crop_kind, bbox_margin)
        out = np.zeros_like(video_frames_rgb[t])
        out[keep] = video_frames_rgb[t][keep]
        frames.append(out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mpy.ImageSequenceClip(frames, fps=fps).write_videofile(out_path)
