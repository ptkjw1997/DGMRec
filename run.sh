#!/bin/bash
# One-command reproduction entrypoint.
#   run.sh <MODEL> <DATASET> [SEED]
# MODEL   ∈ {DGMRec, LGMRec, GUME, DAMRS, MGCN, BM3, LATTICE, SLMRec, GRCN,
#            MMGCN, VBPR, MFBPR, NGCF, SGL, SimGCL, LightGCN, MILK, SIBRAR, CI2MG}
# DATASET ∈ {baby, sports, clothing, elec, tiktok}
#           (tiktok features/interactions are not redistributed with this
#            release — see the reproducibility statement in the README for
#            how to obtain and place them under data/tiktok/)
# SEED    defaults to 999; the paper's multi-seed runs use {999, 42, 2023, 2024, 2025}.
set -e

MODEL=${1:?usage: run.sh MODEL DATASET [SEED]}
DATASET=${2:?usage: run.sh MODEL DATASET [SEED]}
SEED=${3:-999}

cd "$(dirname "$0")"
BEST="src/configs/best/${MODEL}_${DATASET}.json"
OVERRIDE="{}"
if [ -f "$BEST" ]; then
    OVERRIDE=$(cat "$BEST")
fi

# missing-feature ratio used in the paper (0.666 Amazon, 0.6 tiktok);
# defaults also live in src/configs/{overall,dataset/*}.yaml
RATIO=0.666
if [ "$DATASET" = "tiktok" ]; then
    RATIO=0.6
fi

cd src
exec python main.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --gpu_id 0 \
    --missing_modal 1 \
    --missing_ratio "$RATIO" \
    --missing_imputation "[1]" \
    --seed "[$SEED]" \
    --config_override "$OVERRIDE"
