#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=diffusion-main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 02:00:00

set -euo pipefail

: "${RUN_DIR:?RUN_DIR must be set by the submit wrapper}"

# Go to your project
cd /home/willzhao/diffusion

source /home/willzhao/myenv/bin/activate
module load miniforge/24.3.0-0
pip install -r /home/willzhao/diffusion/requirements.txt
export PYTHONUNBUFFERED=1

# Create subfolders for this run
mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/metrics" "${RUN_DIR}/samples"

# Make src/ importable as a source root
export PYTHONPATH="/home/willzhao/diffusion/src:${PYTHONPATH:-}"

python /home/willzhao/diffusion/src/main.py --run-dir "${RUN_DIR}"
