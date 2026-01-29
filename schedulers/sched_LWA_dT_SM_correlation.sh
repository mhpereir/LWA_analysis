#!/bin/bash
# Below specify requested resources
#PBS -S /bin/bash
#PBS -N lwa_deltaT_corr
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -j oe
#PBS -o /home/mhpereir/LWA_analysis/logs/
#PBS -J 0-2


export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

SEASON="JJA"
ZG_LEVEL=500
LWA_VARS=(LWA LWA_a LWA_c)
REGIONS=(pnw_bartusek) 
# ==================================


SCRIPT_PATH="/home/mhpereir/LWA_analysis/scripts/LWA_deltaT_SM_correlation.py"
cd /home/mhpereir/LWA_analysis/

# ==================================


# ----- decode 0..8 into (region_idx, var_idx) -----
K=${PBS_ARRAY_INDEX:-0}                       # 0..2
NV=${#LWA_VARS[@]}                            # 3
NR=${#REGIONS[@]}                             # 1
N=$((NV * NR))
if (( K < 0 || K >= N )); then
  echo "[error] PBS_ARRAY_INDEX=$K out of range 0..$((N-1))"; exit 2
fi
region_idx=$(( K / NV ))                      # 0..0
var_idx=$(( K % NV ))                         # 0..2

REGION=${REGIONS[$region_idx]}
LWA_VAR=${LWA_VARS[$var_idx]}


LOGDIR="./logs"; mkdir -p "$LOGDIR"

echo "[info] $(date -Is) starting REGION=${REGION} on host $(hostname)"
/usr/bin/time -v python "$SCRIPT_PATH" --region "$REGION" --season "$SEASON" --lwa_var "$LWA_VAR" --zg "$ZG_LEVEL"

echo "[info] $(date -Is) done REGION=${REGION}"