#!/bin/bash
# Below specify requested resources
#PBS -S /bin/bash
#PBS -N lwa_during_HEs
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -j oe
#PBS -o /home/mhpereir/LWA_analysis/logs/
#PBS -J 0-1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail
SEASONS=(DJF JJA)
REGIONS=(pnw_bartusek)
ZG_LEVEL=500
Q_HOT=95
Q_COLD=5
WHICH="hot"  # hot | cold | both
HW_THRESH_ROOT="/home/mhpereir/HW_analysis/thresholds"
# ==================================

SCRIPT_PATH="/home/mhpereir/LWA_analysis/scripts/LWA_during_HEs.py"
cd /home/mhpereir/LWA_analysis/

# ==================================

# ----- decode 0..N-1 into (region_idx, season_idx) -----
K=${PBS_ARRAY_INDEX:-0}                       # 0..1
NS=${#SEASONS[@]}                             # 2
NR=${#REGIONS[@]}                             # 1
N=$((NS * NR))
if (( K < 0 || K >= N )); then
  echo "[error] PBS_ARRAY_INDEX=$K out of range 0..$((N-1))"; exit 2
fi
region_idx=$(( K / NS ))                      # 0..0
season_idx=$(( K % NS ))                      # 0..1

REGION=${REGIONS[$region_idx]}
SEASON=${SEASONS[$season_idx]}

LOGDIR="./logs"; mkdir -p "$LOGDIR"

echo "[info] $(date -Is) starting REGION=${REGION} SEASON=${SEASON} on host $(hostname)"
/usr/bin/time -v python "$SCRIPT_PATH" \
  --region "$REGION" \
  --season "$SEASON" \
  --zg "$ZG_LEVEL" \
  --q-hot "$Q_HOT" \
  --q-cold "$Q_COLD" \
  --which "$WHICH" \
  --hw-thresh-root "$HW_THRESH_ROOT"

echo "[info] $(date -Is) done REGION=${REGION} SEASON=${SEASON}"
