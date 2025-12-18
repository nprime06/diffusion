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

cd /home/willzhao/diffusion
source /home/willzhao/myenv/bin/activate
module load miniforge/24.3.0-0
pip install -r /home/willzhao/diffusion/requirements.txt

export PYTHONUNBUFFERED=1

python /home/willzhao/diffusion/src/test_arch/test_unet.py