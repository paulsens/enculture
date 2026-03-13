#!/bin/bash
# Launcher for paired pretraining jobs on enculturation/genre datasets.
# Usage: bash scripts/pretrain.sh "description" [--folds 0-0] [--lr 0.00001] [--task CLS_only] [--dataset enc_shanxiSS]
#
# Examples:
#   bash scripts/pretrain.sh "exp1" --folds 0-11 --dataset genre_NTP
#   bash scripts/pretrain.sh "exp1" --folds 0-0 --dataset enc_bachSS --task CLS_only

cd "$(dirname "$0")" || exit

MSG="${1:?Please provide a description as the first argument}"
FOLDS="${2:-0-0}"
LR="${3:-0.00001}"
TASK="${4:-CLS_only}"
DATASET="${5:-enc_shanxiSS}"
ATN_HEADS="${6:-2}"
NUM_LAYERS="${7:-3}"
FWD_EXP="${8:-4}"
SEQ_LEN="${9:-5}"

IFS='-' read -r FOLD_START FOLD_END <<< "$FOLDS"
FOLD_END="${FOLD_END:-$FOLD_START}"

for i in $(seq "$FOLD_START" "$FOLD_END"); do
    sbatch pretrain.script "$MSG" "$i" "1" "$LR" "False" "$TASK" "$ATN_HEADS" "$NUM_LAYERS" "$FWD_EXP" "$SEQ_LEN" "$DATASET"
    sleep 2
done
