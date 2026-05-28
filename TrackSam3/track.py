import os
from random import shuffle

import cv2
import numpy as np
from decord import VideoReader


# Minimum mask ratio threshold (percentage of frame)
MIN_MASK_RATIO = 1.0

# Default cap on number of targets when caller does not override
DEFAULT_MAX_TARGETS = 4

# Deterministic BGR palette used when callers want stable colors across runs.
DEFAULT_PALETTE_BGR = [
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (0, 255, 255),  # Yellow
]


def remove_small_tracks_from_predictor(predictor, invalid_track_ids):
    """Remove invalid track IDs from predictor's internal tracker state."""
    if not invalid_track_ids:
        return

    metadata = predictor.inference_state.get("tracker_metadata", {})
    if not metadata:
        return

    obj_ids = metadata.get("obj_ids_all_gpu", np.array([]))
    if len(obj_ids) == 0:
        return

    keep_mask = np.array([int(oid) not in invalid_track_ids for oid in obj_ids])
    metadata["obj_ids_all_gpu"] = obj_ids[keep_mask]

    arrays_to_filter = [
        "obj_id_to_score", "obj_id_to_cls", "obj_id_to_tracker_score"
    ]
    for key in arrays_to_filter:
        if key in metadata and isinstance(metadata[key], dict):
            metadata[key] = {k: v for k, v in metadata[key].items() if int(k) not in invalid_track_ids}

    tracker_states = predictor.inference_state.get("tracker_inference_states", [])
    if tracker_states:
        for state in tracker_states:
            if hasattr(state, 'obj_ids') and state.obj_ids is not None:
                state_keep = np.array([int(oid) not in invalid_track_ids for oid in state.obj_ids])
                state.obj_ids = state.obj_ids[state_keep]

    print(f"Removed track IDs {invalid_track_ids} from tracker state")


def visualize_and_save_mask(results, width, height, predictor, new_indices, full_length,
                            max_targets=DEFAULT_MAX_TARGETS, shuffle_colors=True,
                            direct_return=False):
    """Run through SAM3 streaming results and gather per-track binary masks.

    Returns (valid_track_ids_ordered, mask_arrays, track_colors) ordered by descending
    mask area in the first frame; or None if no valid track is detected.
    """
    colors = list(DEFAULT_PALETTE_BGR)
    if shuffle_colors:
        shuffle(colors)

    frame_idx = 0
    valid_track_ids = None
    total_pixels = height * width
    valid_track_ids_ordered = []
    mask_arrays = {}
    track_colors = {}
    _color_counter = 0

    for result_idx, result in enumerate(results):
        index_result = new_indices[result_idx]
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # (N, H, W)
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else np.arange(len(masks))

            if frame_idx == 0:
                valid_track_ids = set()
                invalid_track_ids = set()
                candidates = []
                for i, (mask, track_id) in enumerate(zip(masks, track_ids)):
                    if mask.shape[:2] != (height, width):
                        mask_resized = cv2.resize(mask.astype(np.float32), (width, height))
                    else:
                        mask_resized = mask
                    mask_bool = mask_resized > 0.5
                    mask_ratio = np.sum(mask_bool) / total_pixels * 100
                    if mask_ratio >= MIN_MASK_RATIO:
                        candidates.append((int(track_id), mask_ratio))
                    else:
                        invalid_track_ids.add(int(track_id))

                candidates.sort(key=lambda x: x[1], reverse=True)

                if len(candidates) == 0 and direct_return:
                    print(f"  No valid candidates (all < MIN_MASK_RATIO={MIN_MASK_RATIO}%) in first frame, return")
                    return

                if len(candidates) > max_targets:
                    if direct_return:
                        print(f"  Found {len(candidates)} candidates, return")
                        return
                    print(f"  Found {len(candidates)} candidates, limiting to top {max_targets}")
                    kept_candidates = candidates[:max_targets]
                    dropped_candidates = candidates[max_targets:]
                    for track_id, _ in kept_candidates:
                        valid_track_ids.add(track_id)
                    for track_id, _ in dropped_candidates:
                        invalid_track_ids.add(track_id)
                else:
                    kept_candidates = candidates
                    for track_id, _ in candidates:
                        valid_track_ids.add(track_id)

                if kept_candidates and direct_return:
                    max_ratio = kept_candidates[0][1]
                    if max_ratio < 1.5 or max_ratio > 50:
                        print(f"  Max mask ratio {max_ratio:.2f}% out of valid range [1.5, 50], return")
                        return

                if len(kept_candidates) >= 2 and direct_return:
                    max_ratio = kept_candidates[0][1]
                    min_ratio = kept_candidates[-1][1]
                    if min_ratio < max_ratio / 3:
                        print(f"  Smallest person ({min_ratio:.2f}%) < 1/3 of largest ({max_ratio:.2f}%), return")
                        return

                valid_track_ids_ordered = [tid for tid, _ in kept_candidates]
                mask_arrays = {tid: np.zeros((full_length, height, width), dtype=bool)
                               for tid in valid_track_ids_ordered}

                if invalid_track_ids:
                    remove_small_tracks_from_predictor(predictor, invalid_track_ids)

            for i, (mask, track_id) in enumerate(zip(masks, track_ids)):
                if valid_track_ids is not None and int(track_id) not in valid_track_ids:
                    continue

                tid = int(track_id)
                if tid not in track_colors:
                    track_colors[tid] = colors[_color_counter % len(colors)]
                    _color_counter += 1

                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask.astype(np.float32), (width, height))
                mask_bool = mask > 0.5

                if tid in mask_arrays:
                    mask_arrays[tid][index_result] = mask_bool

        frame_idx += 1

    if not valid_track_ids_ordered:
        return None
    return valid_track_ids_ordered, mask_arrays, track_colors


def _centroid_x(mask_2d):
    """X-coordinate of the centroid of a 2D bool mask. Returns +inf if mask is empty."""
    cols = np.where(mask_2d.any(axis=0))[0]
    if len(cols) == 0:
        return float('inf')
    rows = np.where(mask_2d.any(axis=1))[0]
    # use bounding-box center (cheap and stable)
    return 0.5 * (cols[0] + cols[-1])


def _reorder_and_color(valid_track_ids_ordered, mask_arrays, sort_by, fixed_colors):
    """Apply left-to-right sort and deterministic color assignment.

    Returns (masks, colors) where masks is a list of (T, H, W) bool ndarray and
    colors is a list of BGR tuples, both in the chosen ordering.
    """
    if sort_by == 'x':
        ordered = sorted(valid_track_ids_ordered,
                         key=lambda tid: _centroid_x(mask_arrays[tid][0]))
    elif sort_by == 'area':
        ordered = list(valid_track_ids_ordered)
    else:
        raise ValueError(f"unknown sort_by: {sort_by}")

    n = len(ordered)
    if fixed_colors is not None:
        if len(fixed_colors) < n:
            raise ValueError(f"fixed_colors has {len(fixed_colors)} entries but {n} tracks")
        colors = [tuple(c) for c in fixed_colors[:n]]
    else:
        colors = [DEFAULT_PALETTE_BGR[i % len(DEFAULT_PALETTE_BGR)] for i in range(n)]

    masks = [mask_arrays[tid] for tid in ordered]
    return masks, colors


def get_mask_from_video(video_path, predictor, max_targets=DEFAULT_MAX_TARGETS,
                        sort_by='area', fixed_colors=None,
                        text=("human", "character")):
    """Run SAM3 tracking on a video file and return per-person binary masks and colors.

    Args:
        video_path: path to input video (str or Path).
        predictor:  SAM3VideoSemanticPredictor instance (state will be reset).
        max_targets: cap on the number of tracked persons (kept by descending area).
        sort_by:    'area' (default, descending area) or 'x' (left-to-right by first-frame
                    centroid x).
        fixed_colors: optional list of BGR tuples assigned to ordered tracks instead of
                    the default palette. Must have at least len(tracks) entries.

    Returns:
        masks:  list of (T, H, W) bool ndarray, one per tracked person.
        colors: list of BGR color tuples corresponding to each person.
        Both lists are empty if no valid persons are detected.
    """
    video_path = str(video_path)

    predictor.inference_state = {}
    if hasattr(predictor, 'dataset'):
        predictor.dataset = None

    vr = VideoReader(video_path)
    full_length = len(vr)
    height, width = vr[0].asnumpy().shape[:2]
    del vr

    results = predictor(source=video_path, text=list(text), stream=True)
    ret = visualize_and_save_mask(
        results, width, height, predictor,
        new_indices=np.arange(full_length), full_length=full_length,
        max_targets=max_targets, shuffle_colors=fixed_colors is None,
        direct_return=False,
    )
    if ret is None:
        return [], []
    valid_track_ids_ordered, mask_arrays, _ = ret
    return _reorder_and_color(valid_track_ids_ordered, mask_arrays, sort_by, fixed_colors)


def get_mask_from_image_via_video(image_path, video_predictor, max_targets=DEFAULT_MAX_TARGETS,
                                  sort_by='x', fixed_colors=None,
                                  text=("human", "character"), n_repeat=4, fps=8):
    """Detect persons in a still image by wrapping it as a tiny mp4 and routing through
    SAM3VideoSemanticPredictor. Workaround for image-mode SAM3 missing small / distant
    subjects that the video pipeline picks up reliably.

    Returns (masks, colors) with each mask shaped (1, H, W) bool — only the first frame
    of the synthetic clip is kept.
    """
    import tempfile
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(tmp_fd)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(tmp_path, fourcc, float(fps), (W, H))
        if not vw.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open {tmp_path}")
        for _ in range(n_repeat):
            vw.write(img)
        vw.release()

        masks, colors = get_mask_from_video(
            tmp_path, video_predictor,
            max_targets=max_targets, sort_by=sort_by,
            fixed_colors=fixed_colors, text=text,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    masks = [m[:1] for m in masks]
    return masks, colors


def get_mask_from_image(image_path, predictor, max_targets=DEFAULT_MAX_TARGETS,
                        sort_by='x', fixed_colors=None,
                        text=("human", "character")):
    """Run SAM3SemanticPredictor (image variant) on a single image.

    Args:
        image_path: path to input image (str or Path).
        predictor:  SAM3SemanticPredictor instance.
        max_targets: cap on number of persons (kept by descending area).
        sort_by:    'x' (default, left-to-right) or 'area'.
        fixed_colors: optional list of BGR tuples assigned in order; otherwise the
                    deterministic palette is used.

    Returns:
        masks:  list of (1, H, W) bool ndarray, one per detected person.
        colors: list of BGR color tuples corresponding to each person.
    """
    image_path = str(image_path)
    results = predictor(source=image_path, text=list(text))
    if not results:
        return [], []
    result = results[0]
    if result.masks is None or len(result.masks) == 0:
        return [], []

    masks_NHW = result.masks.data.cpu().numpy()  # (N, H, W)
    masks_NHW = masks_NHW > 0.5

    return _filter_image_masks(masks_NHW, max_targets, sort_by, fixed_colors)


def _filter_image_masks(masks_NHW, max_targets, sort_by, fixed_colors):
    """Apply MIN_MASK_RATIO + max_targets filter to per-image SAM masks, then order
    and color them. Returns (masks_list, colors) where each mask is (1, H, W) bool.
    """
    N, H, W = masks_NHW.shape
    total_pixels = H * W

    candidates = []  # (idx, mask_ratio)
    for i in range(N):
        ratio = float(np.sum(masks_NHW[i])) / total_pixels * 100
        if ratio >= MIN_MASK_RATIO:
            candidates.append((i, ratio))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:max_targets]
    if not candidates:
        return [], []

    kept = [masks_NHW[idx] for idx, _ in candidates]  # list of (H, W) bool

    if sort_by == 'x':
        order = sorted(range(len(kept)), key=lambda i: _centroid_x(kept[i]))
        kept = [kept[i] for i in order]
    elif sort_by != 'area':
        raise ValueError(f"unknown sort_by: {sort_by}")

    n = len(kept)
    if fixed_colors is not None:
        if len(fixed_colors) < n:
            raise ValueError(f"fixed_colors has {len(fixed_colors)} entries but {n} masks")
        colors = [tuple(c) for c in fixed_colors[:n]]
    else:
        colors = [DEFAULT_PALETTE_BGR[i % len(DEFAULT_PALETTE_BGR)] for i in range(n)]

    masks_out = [m[None] for m in kept]  # add T=1 axis
    return masks_out, colors
