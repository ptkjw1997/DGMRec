#!/bin/bash
set -e

MODEL=${1:?usage: run.sh MODEL DATASET [SEED]}
DATASET=${2:?usage: run.sh MODEL DATASET [SEED]}
SEED=${3:-999}

cd "$(dirname "$0")"
BEST="src/configs/best/${MODEL}/${DATASET}.json"
OVERRIDE="{}"
if [ -f "$BEST" ]; then
    OVERRIDE=$(cat "$BEST")
fi

RATIO=0.666
if [ "$DATASET" = "tiktok" ]; then
    RATIO=0.75
fi

IMPUTATION=2
if [ "$MODEL" = "DGMRec" ]; then
    IMPUTATION=1
fi

cd src
exec python main.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --gpu_id 0 \
    --missing_modal 1 \
    --missing_ratio "$RATIO" \
    --missing_imputation "[$IMPUTATION]" \
    --seed "[$SEED]" \
    --config_override "$OVERRIDE"
