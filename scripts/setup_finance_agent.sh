#!/usr/bin/env bash
# Prepare the Finance-Agent benchmark for use with this repository.
#
# This script:
#   1. Clones the upstream Finance-Agent repository (https://github.com/vals-ai/finance-agent).
#   2. Converts the upstream task definitions into the normalized JSON schema
#      expected by `agent_scaling.datasets.finance_agent.FinanceAgentDataset`.
#   3. Copies the config template into place.
#
# Usage:
#   bash scripts/setup_finance_agent.sh [--upstream-dir <path>] [--out <path>]
#
# After running, edit `run_conf/dataset/finance-agent.yaml` if you placed the
# JSON anywhere other than `datasets/finance_agent.json`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM_DIR="${UPSTREAM_DIR:-${REPO_ROOT}/third_party/finance-agent}"
OUT_JSON="${OUT_JSON:-${REPO_ROOT}/datasets/finance_agent.json}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upstream-dir) UPSTREAM_DIR="$2"; shift 2;;
    --out)          OUT_JSON="$2"; shift 2;;
    *)              echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

mkdir -p "$(dirname "${UPSTREAM_DIR}")"
mkdir -p "$(dirname "${OUT_JSON}")"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  echo "[setup_finance_agent] cloning upstream Finance-Agent into ${UPSTREAM_DIR}"
  git clone --depth=1 https://github.com/vals-ai/finance-agent.git "${UPSTREAM_DIR}"
else
  echo "[setup_finance_agent] upstream already cloned at ${UPSTREAM_DIR}"
fi

echo "[setup_finance_agent] converting upstream tasks -> ${OUT_JSON}"
python "${REPO_ROOT}/scripts/_convert_finance_agent.py" \
  --upstream-dir "${UPSTREAM_DIR}" \
  --out "${OUT_JSON}"

CFG_TEMPLATE="${REPO_ROOT}/run_conf/dataset/finance-agent.yaml.template"
CFG_TARGET="${REPO_ROOT}/run_conf/dataset/finance-agent.yaml"
if [[ ! -f "${CFG_TARGET}" ]]; then
  echo "[setup_finance_agent] writing ${CFG_TARGET} from template"
  cp "${CFG_TEMPLATE}" "${CFG_TARGET}"
fi

echo "[setup_finance_agent] done. Normalized JSON: ${OUT_JSON}"
echo "[setup_finance_agent] you can now run experiments with dataset=finance-agent"
