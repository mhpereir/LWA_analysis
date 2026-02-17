#!/bin/bash
# Below specify requested resources
#PBS -S /bin/bash
#PBS -N bayesian_bothlwa_deltaT_corr_2
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -j oe
#PBS -o /home/mhpereir/LWA_analysis/logs/
#### PBS -J 0-2


export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

SEASON="JJA"
ZG_LEVEL=500
REGIONS=(pnw_bartusek)
BAYESIAN_MODEL="both_lwa_ar1_normal" 

#"both_lwa_noar_studentt"
# ==================================


SCRIPT_PATH="/home/mhpereir/LWA_analysis/scripts/Bayesian_bothLWA_dT_SM_correlation_2.py"
cd /home/mhpereir/LWA_analysis/

# ==================================


# ----- decode 0..8 into (region_idx, var_idx) -----
K=${PBS_ARRAY_INDEX:-0}                       # 0..2
REGION=${REGIONS[$K]}


echo "[info] $(date -Is) starting REGION=${REGION} on host $(hostname)"
/usr/bin/time -v python "$SCRIPT_PATH" --bayesian_model "$BAYESIAN_MODEL" --region "$REGION" --season "$SEASON"  --zg "$ZG_LEVEL"

echo "[info] $(date -Is) done REGION=${REGION}"