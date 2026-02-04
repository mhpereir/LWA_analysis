#!/bin/bash
# Below specify requested resources
#PBS -S /bin/bash
#PBS -N lwa_baseline
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -j oe
#PBS -o /home/mhpereir/LWA_analysis/logs/

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail
ZG_LEVEL=500
HATCH_K=2.0
# ==================================

SCRIPT_PATH="/home/mhpereir/LWA_analysis/scripts/LWA_baseline.py"
cd /home/mhpereir/LWA_analysis

# ==================================

echo "[info] $(date -Is) starting LWA_baseline on host $(hostname)"


/usr/bin/time -v python "$SCRIPT_PATH" \
  --zg "$ZG_LEVEL" \
  "${EXTRA_ARGS[@]}"

echo "[info] $(date -Is) done"
