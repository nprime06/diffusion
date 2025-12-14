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

# Load python (ORCD recipe uses miniforge)
module load miniforge/24.3.0-0

# Use your existing venv
source /home/willzhao/myenv/bin/activate

# Helpful for seeing prints immediately in output files
export PYTHONUNBUFFERED=1

# Run training (edit to match your entry point)
python /home/willzhao/diffusion/src/test_unet.py
