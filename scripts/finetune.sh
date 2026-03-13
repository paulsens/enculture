#!/bin/bash
# Launcher for paired finetuning jobs with pretrained model transfer.
# Usage: bash scripts/finetune.sh "description" [--folds 0-11] [--lr 0.00001] [--task CLS_only] [--dataset enc_shanxiSS] [--pretrain_task enc_bachSS]
#
# Examples:
#   bash scripts/finetune.sh "exp1" --folds 0-11 --dataset genre_NTP --pretrain_task genre_NTP
#   bash scripts/finetune.sh "exp1" --dataset enc_shanxiSS --pretrain_task enc_bachSS

cd "$(dirname "$0")" || exit

MSG="${1:?Please provide a description as the first argument}"
FOLDS="${2:-0-11}"
LR="${3:-0.00001}"
TASK="${4:-CLS_only}"
DATASET="${5:-enc_shanxiSS}"
PRETRAIN_TASK="${6:-enc_bachSS}"
ATN_HEADS="${7:-2}"
NUM_LAYERS="${8:-3}"
FWD_EXP="${9:-4}"
SEQ_LEN="${10:-5}"

IFS='-' read -r FOLD_START FOLD_END <<< "$FOLDS"
FOLD_END="${FOLD_END:-$FOLD_START}"

for i in $(seq "$FOLD_START" "$FOLD_END"); do
    sbatch finetune.script "$MSG" "$i" "1" "$LR" "False" "$TASK" "$ATN_HEADS" "$NUM_LAYERS" "$FWD_EXP" "$SEQ_LEN" "$DATASET" "$PRETRAIN_TASK"
    sleep 2
done
