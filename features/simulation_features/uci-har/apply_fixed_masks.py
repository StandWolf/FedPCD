# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""Apply the fixed UCI-HAR client-to-modality assignment.

``simulation_feature.py`` writes the sample/label structure of the simulation
files. The training-client modality assignment used in the paper is fixed and
stored in ``fixed_masks_alpha005.json`` (one entry per training client, with
``[acc_missing, gyro_missing]``). This script applies that assignment to the
generated files so that the masks do not depend on the NumPy version.

``dev`` and ``test`` are evaluated with complete inputs by the training code,
so their entries are left untouched.

Run from the repository root:

    python features/simulation_features/uci-har/apply_fixed_masks.py --alpha 0.05
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_dir", type=str,
                        default="output/simulation_feature/uci-har")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--table", type=str,
                        default=str(HERE / "fixed_masks_alpha005.json"))
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir)
    alpha_str = str(args.alpha).replace(".", "")
    with open(args.table) as f:
        table = json.load(f)

    for tag, assignment in table.items():
        path = sim_dir / f"{tag}_alpha{alpha_str}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run simulation_feature.py first")
        with open(path) as f:
            data = json.load(f)

        missing = [cid for cid in assignment if cid not in data]
        if missing:
            raise KeyError(f"{tag}: clients absent from {path}: {missing[:5]}")

        for cid, mask in assignment.items():
            if cid in ("dev", "test"):
                continue
            for entry in data[cid]:
                entry[-1][0], entry[-1][1] = mask

        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"applied fixed masks to {path}")


if __name__ == "__main__":
    main()
