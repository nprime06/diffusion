#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=unet-test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 02:00:00
#SBATCH --output=/home/willzhao/diffusion/logs/test_unet/%x-%j.out
#SBATCH --error=/home/willzhao/diffusion/logs/test_unet/%x-%j.err

set -euo pipefail

# Go to your project
cd /home/willzhao/diffusion

source /home/willzhao/myenv/bin/activate

module load miniforge/24.3.0-0

pip install -r /home/willzhao/diffusion/requirements.txt

export PYTHONUNBUFFERED=1

python /home/willzhao/diffusion/src/test_arch/test_unet.py


ROOT_DIR="/home/willzhao/diffusion"

METHOD="ddpm"
BACKBONE="unet"

# When run locally, create the run dir and submit THIS script.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  ts="$(date +%Y-%m-%d_%H-%M-%S)"
  run_dir="${ROOT_DIR}/logs/${METHOD}/${METHOD}_${BACKBONE}_${ts}"

  mkdir -p "${run_dir}"

  sbatch \
    --output="${run_dir}/%x-%j.log" \
    --error="${run_dir}/%x-%j.err" \
    --export=ALL,RUN_DIR="${run_dir}",METHOD="${METHOD}",BACKBONE="${BACKBONE}" \
    "${BASH_SOURCE[0]}"

  echo "Submitted job. RUN_DIR=${run_dir}"
  exit 0
fi

: "${RUN_DIR:?RUN_DIR must be set (passed via sbatch --export)}"

cd "${ROOT_DIR}"

mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/metrics" "${RUN_DIR}/samples"

# Copying the same setup style as scripts/test_unet.sh
source /home/willzhao/myenv/bin/activate
module load miniforge/24.3.0-0
pip install -r "${ROOT_DIR}/requirements.txt"
export PYTHONUNBUFFERED=1

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

python "${ROOT_DIR}/src/main.py" --run-dir "${RUN_DIR}"
