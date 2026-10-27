#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
# Download the three public HAR datasets into ./data.
# Usage: bash scripts/download_data.sh [uci-har|mhealth|ku-har|all]
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=${1:-all}
mkdir -p data

fetch() {
    local url=$1 out=$2
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 -o "$out" "$url"
    else
        wget -O "$out" "$url"
    fi
}

download_uci_har() {
    if [ -d data/uci-har/train ]; then
        echo "[uci-har] already present"
        return
    fi
    fetch "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip" data/uci_har.zip
    unzip -q -o data/uci_har.zip -d data
    rm -f data/uci_har.zip
    mv "data/UCI HAR Dataset" data/uci-har
    echo "[uci-har] done"
}

download_mhealth() {
    if [ -d data/mhealth/MHEALTHDATASET ]; then
        echo "[mhealth] already present"
        return
    fi
    mkdir -p data/mhealth
    fetch "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip" data/mhealth.zip
    unzip -q -o data/mhealth.zip -d data/mhealth
    rm -f data/mhealth.zip
    echo "[mhealth] done"
}

download_ku_har() {
    if [ -d data/ku-har/Trimmed_interpolated_data ]; then
        echo "[ku-har] already present"
        return
    fi
    mkdir -p data/ku-har
    fetch "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/45f952y38r-5.zip" data/ku_har.zip
    unzip -q -o data/ku_har.zip -d data/ku-har
    rm -f data/ku_har.zip
    if [ -f data/ku-har/2.Trimmed_interpolated_data.zip ]; then
        unzip -q -o data/ku-har/2.Trimmed_interpolated_data.zip -d data/ku-har
    fi
    echo "[ku-har] done"
}

case "$TARGET" in
    uci-har) download_uci_har ;;
    mhealth) download_mhealth ;;
    ku-har)  download_ku_har ;;
    all)
        download_uci_har
        download_mhealth
        download_ku_har
        ;;
    *) echo "unknown target: $TARGET" >&2; exit 1 ;;
esac

echo "Raw data is under $(pwd)/data"
