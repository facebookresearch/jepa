#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
UCF_ROOT="${UCF_ROOT:-${REPO_ROOT}/data/UCF101}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/outputs}"
SPLITS_DIR="${SPLITS_DIR:-${REPO_ROOT}/data/ucfTrainTestlist}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-60}"
TEST_PER_CLASS="${TEST_PER_CLASS:-20}"
SPLIT_ID="${SPLIT_ID:-1}"
SEED="${SEED:-42}"
NUM_FRAMES="${NUM_FRAMES:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-MCG-NJU/videomae-base}"

CLASSES=(
  ApplyEyeMakeup
  Basketball
  Biking
  Diving
  WalkingWithDog
)

cd "${REPO_ROOT}"

"${PYTHON}" "${SCRIPT_DIR}/make_ucf_subset.py" \
  --ucf_root "${UCF_ROOT}" \
  --official_splits_dir "${SPLITS_DIR}" \
  --out_dir "${OUTPUT_ROOT}/data/ucf101_subset" \
  --classes "${CLASSES[@]}" \
  --split_id "${SPLIT_ID}" \
  --train_per_class "${TRAIN_PER_CLASS}" \
  --test_per_class "${TEST_PER_CLASS}" \
  --seed "${SEED}"

"${PYTHON}" "${SCRIPT_DIR}/extract_frames.py" \
  --metadata "${OUTPUT_ROOT}/data/ucf101_subset/metadata.csv" \
  --out_dir "${OUTPUT_ROOT}/data/frames" \
  --num_frames "${NUM_FRAMES}" \
  --image_size "${IMAGE_SIZE}"

"${PYTHON}" "${SCRIPT_DIR}/extract_videomae_features.py" \
  --frames_metadata "${OUTPUT_ROOT}/data/frames_metadata.csv" \
  --out_dir "${OUTPUT_ROOT}/features/videomae" \
  --model_name "${VIDEOMAE_MODEL}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}"

"${PYTHON}" "${SCRIPT_DIR}/train_linear_probe.py" \
  --features_dir "${OUTPUT_ROOT}/features/videomae" \
  --model_name videomae \
  --out_dir "${OUTPUT_ROOT}/results/videomae" \
  --classifier logistic_regression \
  --max_iter 2000 \
  --run_config "${EXPERIMENT_DIR}/configs/default.yaml" \
  --device_used "${DEVICE}" \
  --num_frames "${NUM_FRAMES}" \
  --image_size "${IMAGE_SIZE}"

printf 'VideoMAE baseline complete: %s\n' "${OUTPUT_ROOT}/results/videomae"
