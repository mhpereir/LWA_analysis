#!/bin/bash
# Below specify requested resources
#PBS -S /bin/bash
#PBS -N lwa_phase_plot
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -j oe
#PBS -o /home/mhpereir/LWA_analysis/logs/
#PBS -J 0-5


export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env


set -euo pipefail

SEASONS=(DJF JJA)
ZG_LEVEL=500
LAT_BANDS=(north south all) 
# ==================================

SCRIPT_PATH="/home/mhpereir/LWA_analysis/scripts/LWA_phase.py"
cd /home/mhpereir/LWA_analysis

# ==================================


# ----- decode 0..5 into (region_idx, var_idx) -----
K=${PBS_ARRAY_INDEX:-0}                       # 0..5
NV=${#SEASONS[@]}                             # 2
NR=${#LAT_BANDS[@]}                             # 3
N=$((NV * NR))
if (( K < 0 || K >= N )); then
  echo "[error] PBS_ARRAY_INDEX=$K out of range 0..$((N-1))"; exit 2
fi
region_idx=$(( K / NV ))                      # 0..2
var_idx=$(( K % NV ))                         # 0..2

LAT_BAND=${LAT_BANDS[$region_idx]}
SEASON=${SEASONS[$var_idx]}

echo "[info] $(date -Is) starting LAT_BAND=${LAT_BAND} on host $(hostname)"
/usr/bin/time -v python "$SCRIPT_PATH" --lat_band "$LAT_BAND" --season "$SEASON" --zg "$ZG_LEVEL"

echo "[info] $(date -Is) done LAT_BAND=${LAT_BAND}"