#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
UCF_ROOT="${UCF_ROOT:-${REPO_ROOT}/data/UCF101}"
SPLITS_DIR="${SPLITS_DIR:-${REPO_ROOT}/data/ucfTrainTestlist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_DIR}/outputs}"
REPORT_DIR="${REPORT_DIR:-${EXPERIMENT_DIR}/reports/latest}"
VJEPA_CONFIG="${VJEPA_CONFIG:-${REPO_ROOT}/configs/pretrain/vitl16.yaml}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-${REPO_ROOT}/checkpoints/vitl16.pth.tar}"
VIDEOMAE_MODEL="${VIDEOMAE_MODEL:-MCG-NJU/videomae-base}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-60}"
TEST_PER_CLASS="${TEST_PER_CLASS:-20}"
NUM_FRAMES="${NUM_FRAMES:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
SPLIT_IDS="${SPLIT_IDS:-1 2 3}"
export HF_HOME="${HF_HOME:-${OUTPUT_ROOT}/cache/huggingface}"

CLASSES=(ApplyEyeMakeup Basketball Biking Diving WalkingWithDog)

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python environment not found: %s\n' "${PYTHON}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/data" "${OUTPUT_ROOT}/features" \
  "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/cache/features" "${REPORT_DIR}"

cd "${REPO_ROOT}"
for split_id in ${SPLIT_IDS}; do
  split_name="$(printf 'split_%02d' "${split_id}")"
  data_dir="${OUTPUT_ROOT}/data/${split_name}"
  metadata="${data_dir}/metadata.csv"
  frames_metadata="${data_dir}/frames_metadata.csv"
  run_config="${data_dir}/run_config.yaml"

  "${PYTHON}" "${SCRIPT_DIR}/make_ucf_subset.py" \
    --ucf_root "${UCF_ROOT}" \
    --official_splits_dir "${SPLITS_DIR}" \
    --out_dir "${data_dir}" \
    --classes "${CLASSES[@]}" \
    --split_id "${split_id}" \
    --train_per_class "${TRAIN_PER_CLASS}" \
    --test_per_class "${TEST_PER_CLASS}" \
    --seed "${SEED}"

  "${PYTHON}" "${SCRIPT_DIR}/extract_frames.py" \
    --metadata "${metadata}" \
    --out_dir "${OUTPUT_ROOT}/data/frames" \
    --metadata_out "${frames_metadata}" \
    --num_frames "${NUM_FRAMES}" \
    --image_size "${IMAGE_SIZE}"

  "${PYTHON}" "${SCRIPT_DIR}/write_run_config.py" \
    --metadata "${metadata}" \
    --output "${run_config}" \
    --split_id "${split_id}" \
    --seed "${SEED}" \
    --num_frames "${NUM_FRAMES}" \
    --image_size "${IMAGE_SIZE}" \
    --device "${DEVICE}" \
    --vjepa_config "${VJEPA_CONFIG}" \
    --vjepa_checkpoint "${VJEPA_CHECKPOINT}" \
    --videomae_model "${VIDEOMAE_MODEL}"

  "${PYTHON}" "${SCRIPT_DIR}/extract_videomae_features.py" \
    --frames_metadata "${frames_metadata}" \
    --out_dir "${OUTPUT_ROOT}/features/${split_name}/videomae" \
    --cache_dir "${OUTPUT_ROOT}/cache/features/videomae_mean_16x224" \
    --model_name "${VIDEOMAE_MODEL}" \
    --device "${DEVICE}" \
    --batch_size 1

  "${PYTHON}" "${SCRIPT_DIR}/extract_vjepa_features.py" \
    --frames_metadata "${frames_metadata}" \
    --out_dir "${OUTPUT_ROOT}/features/${split_name}/vjepa" \
    --cache_dir "${OUTPUT_ROOT}/cache/features/vjepa_vitl16_mean_16x224" \
    --config "${VJEPA_CONFIG}" \
    --checkpoint "${VJEPA_CHECKPOINT}" \
    --device "${DEVICE}" \
    --batch_size 1 \
    --pooling mean

  for model in vjepa videomae; do
    "${PYTHON}" "${SCRIPT_DIR}/train_linear_probe.py" \
      --features_dir "${OUTPUT_ROOT}/features/${split_name}/${model}" \
      --model_name "${model}" \
      --out_dir "${OUTPUT_ROOT}/results/${split_name}/${model}" \
      --classifier logistic_regression \
      --max_iter 2000 \
      --run_config "${run_config}" \
      --device_used "${DEVICE}" \
      --num_frames "${NUM_FRAMES}" \
      --image_size "${IMAGE_SIZE}"
  done
done

"${PYTHON}" "${SCRIPT_DIR}/aggregate_results.py" \
  --results_root "${OUTPUT_ROOT}/results" \
  --split_ids ${SPLIT_IDS}

split_one_frames="${OUTPUT_ROOT}/data/split_01/frames_metadata.csv"
if [[ -f "${split_one_frames}" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/generate_temporal_progression.py" \
    --frames_metadata "${split_one_frames}" \
    --results_root "${OUTPUT_ROOT}/results" \
    --work_dir "${OUTPUT_ROOT}/cache/temporal_progression" \
    --out_dir "${REPORT_DIR}" \
    --vjepa_config "${VJEPA_CONFIG}" \
    --vjepa_checkpoint "${VJEPA_CHECKPOINT}" \
    --videomae_model "${VIDEOMAE_MODEL}" \
    --device "${DEVICE}" \
    --examples 3
fi

"${PYTHON}" "${SCRIPT_DIR}/generate_visual_report.py" \
  --results_root "${OUTPUT_ROOT}/results" \
  --features_root "${OUTPUT_ROOT}/features" \
  --frames_metadata "${split_one_frames}" \
  --out_dir "${REPORT_DIR}" \
  --split_ids ${SPLIT_IDS}

cp "${OUTPUT_ROOT}/results/aggregate/"*.csv "${REPORT_DIR}/"
cp "${OUTPUT_ROOT}/results/aggregate/"*.md "${REPORT_DIR}/"
cp "${OUTPUT_ROOT}/results/aggregate/summary.txt" "${REPORT_DIR}/"

printf 'Benchmark complete. Report: %s/report.html\n' "${REPORT_DIR}"
