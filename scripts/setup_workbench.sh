#!/usr/bin/env bash
# Prepare the WorkBench benchmark for use with this repository.
#
# This script:
#   1. Clones the upstream WorkBench repository (https://github.com/olly-styles/WorkBench).
#   2. Converts the upstream task definitions into the normalized JSON schema
#      expected by `agent_scaling.datasets.workbench.WorkbenchDataset`.
#   3. Copies the config template into place.
#
# Usage:
#   bash scripts/setup_workbench.sh [--upstream-dir <path>] [--out <path>]
#
# After running, edit `run_conf/dataset/workbench.yaml` if you placed the
# JSON anywhere other than `datasets/workbench.json`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-${REPO_ROOT}/third_party/WorkBench}"
OUT_JSON="${OUT_JSON:-${REPO_ROOT}/datasets/workbench.json}"
FULL_OUT_JSON="${FULL_OUT_JSON:-${REPO_ROOT}/datasets/workbench_full_690.json}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
SEED="${SEED:-42}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upstream-dir) UPSTREAM_DIR="$2"; shift 2;;
    --out)          OUT_JSON="$2"; shift 2;;
    --full-out)     FULL_OUT_JSON="$2"; shift 2;;
    --sample-size)  SAMPLE_SIZE="$2"; shift 2;;
    --seed)         SEED="$2"; shift 2;;
    *)              echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

mkdir -p "$(dirname "${UPSTREAM_DIR}")"
mkdir -p "$(dirname "${OUT_JSON}")"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  echo "[setup_workbench] cloning upstream WorkBench into ${UPSTREAM_DIR}"
  git clone --depth=1 https://github.com/olly-styles/WorkBench.git "${UPSTREAM_DIR}"
else
  echo "[setup_workbench] upstream already cloned at ${UPSTREAM_DIR}"
fi

echo "[setup_workbench] converting upstream tasks -> ${OUT_JSON} (stratified ${SAMPLE_SIZE}-instance subset, seed ${SEED})"
echo "[setup_workbench] full upstream JSON also written to -> ${FULL_OUT_JSON}"
python "${REPO_ROOT}/scripts/_convert_workbench.py" \
  --upstream-dir "${UPSTREAM_DIR}" \
  --out "${OUT_JSON}" \
  --full-out "${FULL_OUT_JSON}" \
  --sample-size "${SAMPLE_SIZE}" \
  --seed "${SEED}"

CFG_TEMPLATE="${REPO_ROOT}/run_conf/dataset/workbench.yaml.template"
CFG_TARGET="${REPO_ROOT}/run_conf/dataset/workbench.yaml"
if [[ ! -f "${CFG_TARGET}" ]]; then
  echo "[setup_workbench] writing ${CFG_TARGET} from template"
  cp "${CFG_TEMPLATE}" "${CFG_TARGET}"
fi

echo "[setup_workbench] done. Normalized JSON: ${OUT_JSON}"
echo "[setup_workbench] you can now run experiments with dataset=workbench"
