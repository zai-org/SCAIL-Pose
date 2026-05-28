#!/usr/bin/env bash
# Batch e2e processing for /workspace/ywh_data/EvalCross/cross_pair_gemini.
# Per subdir produces rendered_v2.mp4 (= GT.mp4 copy), rendered_mask_v2.mp4, ref_mask.jpg.
# SAM3 models are loaded only once (batch mode).

cd "$(dirname "$(readlink -f "$0")")"

python NLFPoseExtract/process_animation_aio.py \
    --input_root /workspace/ywh_data/EvalCross/cross_pair_gemini \
    --video_name GT.mp4 \
    --max_persons 2 \
    --e2e_mode \
    "$@"
