#!/usr/bin/env bash
# Batch e2e processing for /workspace/ywh_data/EvalCross/product_eval_long.
# Each subdir (001/002/003/...) has GT.mp4 + ref_image.png|ref.png + meta/.
# Per subdir produces rendered_v2.mp4 (= GT.mp4 copy), rendered_mask_v2.mp4, ref_mask.jpg.
# SAM3 models are loaded only once (batch mode).

cd "$(dirname "$(readlink -f "$0")")"

python NLFPoseExtract/process_animation_aio.py \
    --input_root /workspace/ywh_data/EvalCross/product_eval_long \
    --video_name raw.mp4 \
    --max_persons 2 \
    --e2e_mode \
    "$@"
