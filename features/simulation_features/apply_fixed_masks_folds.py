# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""Apply the fixed client-to-modality assignment for mHealth or KU-HAR.

The structure of each ``mmXX_cl.json`` file (samples, labels, client list) is
produced by the dataset's ``generate_clientlevel_masks.py``. The modality
assignment used in the paper is fixed and stored in ``fixed_masks.json``
(fold -> rate -> client -> ``[acc_missing, gyro_missing]``). This script
overwrites the masks with that assignment so the result is independent of the
NumPy random generator version.

Run from the repository root:

    python features/simulation_features/apply_fixed_masks_folds.py --dataset mhealth
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mhealth", "ku-har"])
    parser.add_argument("--sim_dir", type=str, default=None)
    parser.add_argument("--table", type=str, default=None)
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir) if args.sim_dir else Path("output/simulation_feature") / args.dataset
    table_path = Path(args.table) if args.table else HERE / args.dataset / "fixed_masks.json"

    with open(table_path) as f:
        table = json.load(f)

    for fold, rates in table.items():
        for tag, assignment in rates.items():
            path = sim_dir / fold / f"{tag}_cl.json"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found; run generate_clientlevel_masks.py first")
            with open(path) as f:
                data = json.load(f)

            missing = [cid for cid in assignment if cid not in data]
            if missing:
                raise KeyError(f"{path}: clients absent: {missing[:5]}")

            for cid, mask in assignment.items():
                for entry in data[cid]:
                    entry[-1][0], entry[-1][1] = mask

            with open(path, "w") as f:
                json.dump(data, f)
            print(f"applied fixed masks to {path}")


if __name__ == "__main__":
    main()
