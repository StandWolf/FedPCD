#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
# Ablations reported in the paper.
#
# Usage:
#   bash scripts/run_ablation.sh <dataset> high
#   bash scripts/run_ablation.sh mhealth loss
#   bash scripts/run_ablation.sh <dataset> proj
#
# Environment: PYTHON, FOLDS (default 3), LOGDIR (default logs/ablation)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
DATASET=${1:?usage: run_ablation.sh <uci-har|mhealth|ku-har> <high|loss|proj>}
KIND=${2:-high}
FOLDS=${FOLDS:-3}
LOGDIR=${LOGDIR:-logs/ablation}
mkdir -p "$LOGDIR"

case "$DATASET" in
    uci-har)
        TRAIN=experiment/uci-har/train.py
        BASE="--dataset uci-har --alpha 0.05 --hid_size 128 --sample_rate 0.1 --learning_rate 0.05 --num_epochs 200 --local_epochs 1 --batch_size 16 --missing_protocol clientlevel --num_folds $FOLDS"
        ;;
    mhealth)
        TRAIN=experiment/mhealth/train.py
        BASE="--dataset mhealth --hid_size 128 --sample_rate 0.25 --learning_rate 0.01 --global_learning_rate 0.005 --num_epochs 200 --local_epochs 1 --batch_size 16 --missing_protocol clientlevel --num_folds $FOLDS"
        ;;
    ku-har)
        TRAIN=experiment/ku-har/train.py
        BASE="--dataset ku-har --hid_size 128 --sample_rate 0.1 --learning_rate 0.05 --global_learning_rate 0.025 --num_epochs 200 --local_epochs 1 --batch_size 16 --missing_protocol clientlevel --num_folds $FOLDS"
        ;;
    *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

DISTILL="--en_distill --distill_alpha_max 0.7 --distill_init_T 5.0 --distill_min_T 1.0 --distill_strategy linear --distill_gamma 0.2 --en_double_distill --alpha_feat 0.2"
PCD="$DISTILL --en_att --att_name fuse_base --lambda_rec 1.0 --lambda_proto 0.05 --en_proto_cross --proto_ema 0.97 --proto_temperature 0.07 --lf_proto_support_ref 32"
PCD_WO_PCA="$DISTILL --en_att --att_name fuse_base --lambda_rec 1.0 --leakfree_components rec --lf_proto_support_ref 32"

run_high() {
    for mm in 0.5 0.7 0.8 1.0; do
        for cfg in pcd_wo_pca pcd_full; do
            local flags=$PCD_WO_PCA
            if [ "$cfg" = "pcd_full" ]; then
                flags=$PCD
            fi
            echo "[high] $DATASET $cfg m=$mm"
            "$PYTHON" "$TRAIN" $BASE --fed_alg fed_avg \
                --en_missing_modality --missing_modailty_rate "$mm" $flags \
                > "$LOGDIR/${DATASET}_${cfg}_mm${mm/./}.log" 2>&1
        done
        echo "[high] $DATASET fedavg m=$mm"
        "$PYTHON" "$TRAIN" $BASE --fed_alg fed_avg \
            --en_missing_modality --missing_modailty_rate "$mm" \
            > "$LOGDIR/${DATASET}_fedavg_mm${mm/./}.log" 2>&1
    done
}

run_loss() {
    # Fine-grained loss ablation on mHealth at m=0.7
    if [ "$DATASET" != "mhealth" ]; then
        echo "loss ablation is reported on mhealth; got $DATASET" >&2
        exit 1
    fi
    mm=0.7
    echo "[loss] mhealth fedavg m=$mm"
    "$PYTHON" "$TRAIN" $BASE --fed_alg fed_avg \
        --en_missing_modality --missing_modailty_rate "$mm" \
        > "$LOGDIR/mhealth_loss_fedavg.log" 2>&1
    for comp in logit feat cross logit+feat logit+cross feat+cross logit+feat+cross; do
        echo "[loss] mhealth components=$comp m=$mm"
        "$PYTHON" "$TRAIN" $BASE --fed_alg fed_avg \
            --en_missing_modality --missing_modailty_rate "$mm" \
            --en_att --att_name fuse_base --en_distill --en_double_distill \
            --distill_alpha_max 0.7 --distill_init_T 5.0 --distill_min_T 1.0 \
            --distill_strategy linear \
            --distill_gamma 0.2 \
            --alpha_feat 0.2 --lambda_cross 0.05 --cross_eta 1.0 \
            --no_leak_free --dd_components "$comp" \
            > "$LOGDIR/mhealth_loss_${comp//+/-}.log" 2>&1
    done
}

run_proj() {
    # Projection-head sharing ablation
    if [ "$DATASET" = "uci-har" ]; then
        mm=0.7
    else
        mm=1.0
    fi
    echo "[proj] $DATASET shared heads m=$mm"
    "$PYTHON" "$TRAIN" $BASE --fed_alg fed_avg \
        --en_missing_modality --missing_modailty_rate "$mm" $PCD \
        > "$LOGDIR/${DATASET}_proj_shared.log" 2>&1
    echo "[proj] $DATASET local heads m=$mm"
    "$PYTHON" "$TRAIN" $BASE --fed_alg fed_avg \
        --en_missing_modality --missing_modailty_rate "$mm" $PCD --no_federated_proj \
        > "$LOGDIR/${DATASET}_proj_local.log" 2>&1
}

case "$KIND" in
    high) run_high ;;
    loss) run_loss ;;
    proj) run_proj ;;
    *) echo "unknown ablation: $KIND" >&2; exit 1 ;;
esac

echo "finished: $DATASET / $KIND"
