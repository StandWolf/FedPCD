#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
# Build KU-HAR partitions, features, and client-level missing-modality masks.
# Usage: PYTHON=/path/to/python bash scripts/prepare_ku_har.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}

echo "== fixed Dirichlet partitions (beta=0.3) =="
mkdir -p output/partition/ku-har
cp -r partitions/ku-har/fold* output/partition/ku-har/

echo "== feature extraction =="
"$PYTHON" features/feature_processing/ku-har/extract_feature.py \
    --raw_data_dir data --output_dir output --dataset ku-har

echo "== client-level missing masks =="
"$PYTHON" features/simulation_features/ku-har/generate_clientlevel_masks.py
"$PYTHON" features/simulation_features/apply_fixed_masks_folds.py --dataset ku-har

echo "KU-HAR is ready under ./output"
