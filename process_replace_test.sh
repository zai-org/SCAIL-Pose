#!/usr/bin/env bash
# Batch replacement pipeline for /workspace/ywh_data/evaluation_hard_cross_new/eval_data_test.
# test_mode: driving first frame used as ref (no ref_image file needed).
# Per subdir produces rendered_v2.mp4, replace_mask.mp4 (white bg), ref_image.png, ref_mask.png.

cd "$(dirname "$(readlink -f "$0")")"

CUDA_VISIBLE_DEVICES=0 python NLFPoseExtract/process_replacement.py \
    --input_root /workspace/ywh_data/evaluation_hard_cross_new/eval_data_test \
    --video_name GT.mp4 \
    --test_mode \
    "$@"
