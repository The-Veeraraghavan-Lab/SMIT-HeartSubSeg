#!/usr/bin/env bash

NPROC_PER_NODE="${NPROC_PER_NODE:-$(python - <<'PYTHON'
import torch
count = torch.cuda.device_count()
print(count if count > 0 else 1)
PYTHON
)}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_smit_segmentation.py --distributed --data_dir "data/AllDatasets" --json_list "${DATASET_JSON:-heartsub_master.json}" --datasets "${DATASET_SPLIT:-breast66_labeled}" --results_dir results --output_dir "${OUTPUT_DIR:-b66_plus_cnc64_bnorm}" --model_name smit --pretrained_model_path "${PRETRAINED_MODEL_PATH:-runs/run1_plus_cnc64_bnorm/model_final.pt}" --out_channels "${OUT_CHANNELS:-10}" --norm_name "${NORM_NAME:-batch}" --orientation "${ORIENTATION:-ALS}" --a_min "${A_MIN:--200}" --a_max "${A_MAX:-300}" --space_x "${SPACE_X:-1.0}" --space_y "${SPACE_Y:-1.0}" --space_z "${SPACE_Z:-3.0}" --roi_x "${ROI_X:-128}" --roi_y "${ROI_Y:-128}" --roi_z "${ROI_Z:-128}" --sw_batch_size "${SW_BATCH_SIZE:-12}"
