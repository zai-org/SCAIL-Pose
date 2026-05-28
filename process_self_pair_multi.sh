#!/usr/bin/env bash
# Batch processing for /workspace/ywh_data/evaluation_multiple_human_v3/eval_data.
# Non-e2e mode: rendered_v2.mp4 is the NLF skeleton render (not a copy of GT.mp4),
# rendered_mask_v2.mp4 is the colored SAM3 mask video, ref_mask.jpg is the ref mask.
# SAM3 / NLF / DWpose models are loaded only once (batch mode).

cd "$(dirname "$(readlink -f "$0")")"

CUDA_VISIBLE_DEVICES=1 python NLFPoseExtract/process_animation_aio.py \
    --input_root /workspace/ywh_data/evaluation_multiple_human_v3/eval_data \
    --video_name GT.mp4 \
    --max_persons 2 \
    "$@"
