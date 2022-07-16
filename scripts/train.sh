#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 python train.py \
--exp moment_video \
--model_config config/model_config.json \
--data_config config/data_config.json \
--batch 16 \
--eval_query_batch 5 \
--neg_bmr_pred_num 15 \
--task VCMR \
--eval_tasks VCMR SVMR VR \
--bmr_allowance 500 \
--max_vcmr_video 10 \
--loss_measure moment_video \
--num_workers 8\
${@:1}
