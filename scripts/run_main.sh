#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
# Main benchmark: three datasets x four missing ratios x six methods.
#
# Usage:
#   bash scripts/run_main.sh <dataset> <method> [ratios]
# Examples:
#   bash scripts/run_main.sh uci-har fedpcd
#   bash scripts/run_main.sh mhealth all "0.5 0.7 0.8 1.0"
#
# Environment:
#   PYTHON   python executable (default: python)
#   FOLDS    number of folds / runs (default: 3)
#   LOGDIR   log directory (default: logs/main)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
DATASET=${1:?usage: run_main.sh <uci-har|mhealth|ku-har> <method|all> [ratios]}
METHOD=${2:-fedpcd}
RATIOS=${3:-${RATIOS:-"0.5 0.7 0.8 1.0"}}
FOLDS=${FOLDS:-3}
LOGDIR=${LOGDIR:-logs/main}
mkdir -p "$LOGDIR"

case "$DATASET" in
    uci-har)
        TRAIN=experiment/uci-har/train.py
        MFCPL=experiment/uci-har/train_mfcpl.py
        FEKT=experiment/uci-har/train_fedmekt.py
        BASE="--dataset uci-har --alpha 0.05 --hid_size 128 --sample_rate 0.1 --learning_rate 0.05 --num_epochs 200 --local_epochs 1 --batch_size 16 --missing_protocol clientlevel --num_folds $FOLDS"
        GLR=0.025
        ;;
    mhealth)
        TRAIN=experiment/mhealth/train.py
        MFCPL=experiment/mhealth/train_mfcpl.py
        FEKT=experiment/mhealth/train_fedmekt.py
        BASE="--dataset mhealth --hid_size 128 --sample_rate 0.25 --learning_rate 0.01 --global_learning_rate 0.005 --num_epochs 200 --local_epochs 1 --batch_size 16 --missing_protocol clientlevel --num_folds $FOLDS"
        GLR=0.005
        ;;
    ku-har)
        TRAIN=experiment/ku-har/train.py
        MFCPL=experiment/ku-har/train_mfcpl.py
        FEKT=experiment/ku-har/train_fedmekt.py
        BASE="--dataset ku-har --hid_size 128 --sample_rate 0.1 --learning_rate 0.05 --global_learning_rate 0.025 --num_epochs 200 --local_epochs 1 --batch_size 16 --missing_protocol clientlevel --num_folds $FOLDS"
        GLR=0.025
        ;;
    *)
        echo "unknown dataset: $DATASET" >&2
        exit 1
        ;;
esac

DISTILL="--en_distill --distill_alpha_max 0.7 --distill_init_T 5.0 --distill_min_T 1.0 --distill_strategy linear --distill_gamma 0.2 --en_double_distill --alpha_feat 0.2"
PCD="$DISTILL --en_att --att_name fuse_base --lambda_rec 1.0 --lambda_proto 0.05 --en_proto_cross --proto_ema 0.97 --proto_temperature 0.07 --lf_proto_support_ref 32"

run_one() {
    local method=$1 mm=$2
    local log="$LOGDIR/${DATASET}_${method}_mm${mm/./}.log"
    local cmd

    case "$method" in
        fedavg)
            cmd="$PYTHON $TRAIN $BASE --fed_alg fed_avg --en_missing_modality --missing_modailty_rate $mm"
            ;;
        fedprox)
            cmd="$PYTHON $TRAIN $BASE --fed_alg fed_prox --en_missing_modality --missing_modailty_rate $mm"
            ;;
        fedopt)
            cmd="$PYTHON $TRAIN $BASE --fed_alg fed_opt --global_learning_rate $GLR --en_missing_modality --missing_modailty_rate $mm"
            ;;
        fedpcd)
            cmd="$PYTHON $TRAIN $BASE --fed_alg fed_avg --en_missing_modality --missing_modailty_rate $mm $PCD"
            ;;
        mfcpl)
            cmd="$PYTHON $MFCPL $BASE --en_missing_modality --missing_modailty_rate $mm"
            ;;
        fedmekt)
            cmd="$PYTHON $FEKT $BASE --en_missing_modality --missing_modailty_rate $mm --server_kd_steps 10 --server_kd_lr 0.01 --server_kd_T 3.0 --server_kd_alpha 0.5 --proxy_ratio 0.1"
            ;;
        *)
            echo "unknown method: $method" >&2
            exit 1
            ;;
    esac

    echo "[$(date '+%H:%M:%S')] $DATASET $method m=$mm -> $log"
    eval "$cmd" > "$log" 2>&1
}

if [ "$METHOD" = "all" ]; then
    METHODS="fedavg fedprox fedopt mfcpl fedmekt fedpcd"
else
    METHODS="$METHOD"
fi

for mm in $RATIOS; do
    for m in $METHODS; do
        run_one "$m" "$mm"
    done
done

echo "finished: $DATASET / $METHOD"
