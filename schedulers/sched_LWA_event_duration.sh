#!/bin/bash
# Below specify requested resources
#PBS -S /bin/bash
#PBS -N lwa_deltaT_corr
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -j oe
#PBS -o /home/mhpereir/LWA_analysis/logs/
#PBS -J 0-1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env


set -euo pipefail
SEASONS=(DJF JJA)
ZG_LEVEL=500
LWA_var="LWA_a"
LWA_quantile=95
REGIONS=(pnw_bartusek) # Uncomment for DJF
EVENT_DURATION_THRESHOLD=2 #days (N-1 of consecutive days exceeding threshold)
# ==================================

SCRIPT_PATH="/home/mhpereir/LWA_analysis/scripts/LWA_event_duration.py" #_just_ERA
cd /home/mhpereir/LWA_analysis/

# ==================================

# ----- decode 0..5 into (region_idx, var_idx) -----
K=${PBS_ARRAY_INDEX:-0}                       # 0..1
NV=${#SEASONS[@]}                             # 2
NR=${#REGIONS[@]}                             # 1
N=$((NV * NR))
if (( K < 0 || K >= N )); then
  echo "[error] PBS_ARRAY_INDEX=$K out of range 0..$((N-1))"; exit 2
fi
region_idx=$(( K / NV ))                      # 0..0
var_idx=$(( K % NV ))                         # 0..1

REGION=${REGIONS[$region_idx]}
SEASON=${SEASONS[$var_idx]}

echo "[info] $(date -Is) starting REGION=${REGION} on host $(hostname)"
/usr/bin/time -v python "$SCRIPT_PATH" --lwa_var "$LWA_var" --lwa_quantile "$LWA_quantile" --region "$REGION" --season "$SEASON" --zg "$ZG_LEVEL" --event_duration_threshold "$EVENT_DURATION_THRESHOLD"

echo "[info] $(date -Is) done REGION=${REGION}"
