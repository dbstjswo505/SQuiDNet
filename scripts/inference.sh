#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python inference.py \
--data_config config/data_config.json \
--model_config config/model_config.json \
--max_vcmr_video 10 \
--eval_query_batch 5 \
--nms_thd 1 \
--tasks VCMR SVMR VR \
--model_dir moment_video-2022_07_14_11_16_52 \
${@:1}
