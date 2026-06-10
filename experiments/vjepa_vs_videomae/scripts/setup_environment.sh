#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.12}"

cd "${REPO_ROOT}"
if [[ ! -d .venv ]]; then
  "${PYTHON_BOOTSTRAP}" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-cache-dir -e .
.venv/bin/python -m pip install --no-cache-dir \
  -r "${EXPERIMENT_DIR}/requirements_extra.txt"

printf 'Environment ready: %s\n' "${REPO_ROOT}/.venv"
