#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
# Build UCI-HAR partitions, features, and missing-modality masks.
# Usage: PYTHON=/path/to/python bash scripts/prepare_uci_har.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
ALPHA=${ALPHA:-0.05}

if [ "$ALPHA" != "0.05" ]; then
    echo "the fixed UCI-HAR partition in partitions/ is for alpha=0.05" >&2
    exit 1
fi

echo "== fixed Dirichlet partition (alpha=${ALPHA}, 105 clients) =="
mkdir -p output/partition/uci-har
cp partitions/uci-har/partition_alpha005.json output/partition/uci-har/

echo "== feature extraction =="
"$PYTHON" features/feature_processing/uci-har/extract_feature.py \
    --dataset uci-har --alpha "$ALPHA"

echo "== missing-modality masks (client-level for training clients) =="
for mm in 0.5 0.7 0.8 1.0; do
    "$PYTHON" features/simulation_features/uci-har/simulation_feature.py \
        --dataset uci-har --alpha "$ALPHA" \
        --en_missing_modality --missing_modailty_rate "$mm"
done

echo "== fixed client-to-modality assignment =="
"$PYTHON" features/simulation_features/uci-har/apply_fixed_masks.py \
    --alpha "$ALPHA"

echo "== client-level simulation files =="
"$PYTHON" features/simulation_features/uci-har/make_clientlevel_files.py \
    --alpha "$ALPHA"

echo "UCI-HAR is ready under ./output"
