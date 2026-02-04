#!/bin/bash
# Below specify requested resources
#PBS -S /bin/bash
#PBS -N sm_lwa_relation
#PBS -l select=1:ncpus=8:mem=24gb
#PBS -j oe
#PBS -o /home/mhpereir/LWA_analysis/logs/
#PBS -J 0-2

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

REGIONS=(pnw_bartusek)
LWA_VARS=(LWA LWA_a LWA_c)
# ==================================

SCRIPT_PATH="/home/mhpereir/LWA_analysis/scripts/SM_LWA_relation.py"
cd /home/mhpereir/LWA_analysis

# ==================================

# ----- decode 0..0 into (region_idx) -----
K=${PBS_ARRAY_INDEX:-0}                       # 0..0
NR=${#REGIONS[@]}                             # 1
NV=${#LWA_VARS[@]}                            # 3
N=$((NR * NV))
if (( K < 0 || K >= N )); then
  echo "[error] PBS_ARRAY_INDEX=$K out of range 0..$((N-1))"; exit 2
fi

region_idx=$(( K / NV ))                       # 0..0
lwa_var_idx=$(( K % NV ))                      # 0..2

LWA_VAR=${LWA_VARS[$lwa_var_idx]}
REGION=${REGIONS[$region_idx]}

echo "[info] $(date -Is) starting REGION=${REGION}, LWA_VAR=${LWA_VAR} on host $(hostname)"
/usr/bin/time -v python "$SCRIPT_PATH" --region "$REGION" --lwa_var "$LWA_VAR"

echo "[info] $(date -Is) done REGION=${REGION}, LWA_VAR=${LWA_VAR}"