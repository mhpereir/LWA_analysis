#!/bin/bash
# Below specify requested resources
#PBS -S /bin/bash
#PBS -N testing_bayesian_on_extremes
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -j oe
#PBS -o /home/mhpereir/LWA_analysis/logs/

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail
# ==================================

SCRIPT_PATH="/home/mhpereir/LWA_analysis/scripts/test_bayesian_model_on_extremes.py"
cd /home/mhpereir/LWA_analysis

# ==================================

echo "[info] $(date -Is) starting testing_bayesian_model_on_extremes on host $(hostname)"


/usr/bin/time -v python "$SCRIPT_PATH"

#  \
#   --zg "$ZG_LEVEL" \
#   "${EXTRA_ARGS[@]}"

echo "[info] $(date -Is) done"
