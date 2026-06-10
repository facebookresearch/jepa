#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
UCF_ROOT="${UCF_ROOT:-${DATA_ROOT}/UCF101}"
SPLITS_DIR="${SPLITS_DIR:-${DATA_ROOT}/ucfTrainTestlist}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${DATA_ROOT}/downloads}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

UCF_URL="https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
SPLITS_URL="https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
VJEPA_URL="https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar"

CLASSES=(ApplyEyeMakeup Basketball Biking Diving WalkingWithDog)

download_with_resume() {
  local url="$1"
  local output="$2"
  local attempt
  for attempt in $(seq 1 30); do
    if curl --location --fail --continue-at - -o "${output}" "${url}"; then
      return 0
    fi
    printf 'Download interrupted; resuming attempt %d/30 in 5 seconds.\n' \
      "${attempt}" >&2
    sleep 5
  done
  printf 'Download failed after 30 resumable attempts: %s\n' "${url}" >&2
  return 1
}

available_kb="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
minimum_kb=$((15 * 1024 * 1024))
if (( available_kb < minimum_kb )); then
  printf 'At least 15 GiB free is required before download; available: %.1f GiB\n' \
    "$(awk "BEGIN {print ${available_kb}/1024/1024}")" >&2
  exit 1
fi

mkdir -p "${DATA_ROOT}" "${CHECKPOINT_DIR}" "${DOWNLOAD_DIR}"

if [[ ! -f "${SPLITS_DIR}/trainlist01.txt" ]]; then
  split_archive="${DOWNLOAD_DIR}/UCF101TrainTestSplits.zip"
  download_with_resume "${SPLITS_URL}" "${split_archive}"
  unzip -q -o "${split_archive}" -d "${DATA_ROOT}"
  [[ "${KEEP_ARCHIVES}" == "1" ]] || rm -f "${split_archive}"
fi

missing_class=0
for class_name in "${CLASSES[@]}"; do
  if [[ ! -d "${UCF_ROOT}/${class_name}" ]]; then
    missing_class=1
  fi
done

if [[ "${missing_class}" == "1" ]]; then
  ucf_archive="${DOWNLOAD_DIR}/UCF101.rar"
  download_with_resume "${UCF_URL}" "${ucf_archive}"
  extraction_root="${DATA_ROOT}/ucf_selected_extract"
  rm -rf "${extraction_root}"
  mkdir -p "${extraction_root}"
  archive_paths=()
  for class_name in "${CLASSES[@]}"; do
    archive_paths+=("UCF-101/${class_name}")
  done
  unar -q -f -o "${extraction_root}" "${ucf_archive}" "${archive_paths[@]}"
  mkdir -p "${UCF_ROOT}"
  for class_name in "${CLASSES[@]}"; do
    source_dir="${extraction_root}/UCF-101/${class_name}"
    if [[ ! -d "${source_dir}" ]]; then
      printf 'Could not extract class %s from UCF101 archive.\n' "${class_name}" >&2
      exit 1
    fi
    mv "${source_dir}" "${UCF_ROOT}/${class_name}"
  done
  rm -rf "${extraction_root}"
  [[ "${KEEP_ARCHIVES}" == "1" ]] || rm -f "${ucf_archive}"
fi

checkpoint="${CHECKPOINT_DIR}/vitl16.pth.tar"
if [[ ! -f "${checkpoint}" ]]; then
  download_with_resume "${VJEPA_URL}" "${checkpoint}"
fi

printf 'Assets ready:\n  UCF101: %s\n  splits: %s\n  checkpoint: %s\n' \
  "${UCF_ROOT}" "${SPLITS_DIR}" "${checkpoint}"
