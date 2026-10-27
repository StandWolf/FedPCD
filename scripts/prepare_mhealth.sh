#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
# Build mHealth partitions, features, and client-level missing-modality masks.
# Usage: PYTHON=/path/to/python bash scripts/prepare_mhealth.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}

echo "== fixed Dirichlet partitions (beta=0.1, 15 train clients per fold) =="
mkdir -p output/partition/mhealth
cp -r partitions/mhealth/fold* output/partition/mhealth/

echo "== feature extraction =="
"$PYTHON" features/feature_processing/mhealth/extract_feature.py \
    --raw_data_dir data --output_dir output --dataset mhealth

echo "== client-level missing masks =="
"$PYTHON" features/simulation_features/mhealth/generate_clientlevel_masks.py
"$PYTHON" features/simulation_features/apply_fixed_masks_folds.py --dataset mhealth

echo "mHealth is ready under ./output"
