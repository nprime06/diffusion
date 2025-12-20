#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/willzhao/diffusion"
: "${DATASET:?Set dataset}"
: "${METHOD:?Set method}"
: "${BACKBONE:?Set backbone}"

JOB_SCRIPT="${ROOT_DIR}/scripts/main_job.sh"
if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "Job script not found: ${JOB_SCRIPT}" 1>&2
  exit 1
fi

TS="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="${ROOT_DIR}/logs/${METHOD}/${DATASET}_${METHOD}_${BACKBONE}_${TS}"

mkdir -p "${RUN_DIR}"

sbatch \
  --output="${RUN_DIR}/%x-%j.log" \
  --error="${RUN_DIR}/%x-%j.err" \
  --export=ALL,RUN_DIR="${RUN_DIR}",DATASET="${DATASET}",METHOD="${METHOD}",BACKBONE="${BACKBONE}" \
  "${JOB_SCRIPT}"

echo "Job submitted: RUN_DIR=${RUN_DIR}"