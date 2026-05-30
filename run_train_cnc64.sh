#!/usr/bin/env bash

RESUME_FLAG=()
KEEP_CACHE_FLAG=()
if [[ "${RESUME_LAST_CHECKPOINT:-0}" == "1" ]]; then
  RESUME_FLAG+=(--resume_last_checkpoint)
fi
if [[ "${KEEP_PERSISTENT_CACHE:-0}" == "1" ]]; then
  KEEP_CACHE_FLAG+=(--keep_persistent_cache)
fi

python main_smit.py \
--data_dir "data/AllDatasets" \
--json_list heartsub_master.json \
--train_split set1_cnc64_training \
--val_split set1_cnc64_validation \
--logdir "${LOGDIR:-run1_plus_cnc64_inorm}" \
--dataset_backend "${DATASET_BACKEND:-persistent}" \
--norm_name "${NORM_NAME:-instance}" \
--augmentation_mode "${AUGMENTATION_MODE:-adv}" \
--save_checkpoint \
--max_epochs "${MAX_EPOCHS:-1000}" \
--batch_size "${BATCH_SIZE:-4}" \
--optim_lr "${OPTIM_LR:-2e-4}" \
--val_every "${VAL_EVERY:-50}" \
--checkpoint_save_every "${CHECKPOINT_SAVE_EVERY:-20}" \
--use_ssl_pretrained \
--distributed \
--in_channels 1 \
--out_channels "${OUT_CHANNELS:-10}" \
--a_min -200 \
--a_max 300 \
--space_x "${SPACE_X:-1.0}" \
--space_y "${SPACE_Y:-1.0}" \
--space_z "${SPACE_Z:-3.0}" \
--roi_x 128 \
--roi_y 128 \
--roi_z 128 \
"${RESUME_FLAG[@]}" \
"${KEEP_CACHE_FLAG[@]}"
