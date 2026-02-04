#!/bin/bash
# Below specify requested resources
#PBS -S /bin/bash
#PBS -N sm_baseline
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -j oe
#PBS -o /home/mhpereir/LWA_analysis/logs/
### PBS -J 0-0


export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

REGIONS=(pnw_bartusek)
# ==================================

SCRIPT_PATH="/home/mhpereir/LWA_analysis/scripts/SM_baseline.py"
cd /home/mhpereir/LWA_analysis

# ==================================


# ----- decode 0..0 into (region_idx) -----
K=${PBS_ARRAY_INDEX:-0}                       # 0..0
NR=${#REGIONS[@]}                             # 1
if (( K < 0 || K >= NR )); then
  echo "[error] PBS_ARRAY_INDEX=$K out of range 0..$((NR-1))"; exit 2
fi
region_idx=$(( K ))                           # 0..0

REGION=${REGIONS[$region_idx]}

echo "[info] $(date -Is) starting REGION=${REGION} on host $(hostname)"
/usr/bin/time -v python "$SCRIPT_PATH" --region "$REGION"

echo "[info] $(date -Is) done REGION=${REGION}"
