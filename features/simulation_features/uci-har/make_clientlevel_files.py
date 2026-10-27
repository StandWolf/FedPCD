# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""Create UCI-HAR client-level simulation files.

``apply_fixed_masks.py`` installs the fixed client-to-modality assignment in
``mmXX_alphaYY.json``. This script writes the corresponding
``mmXX_alphaYY_cl.json`` files used when training with
``--missing_protocol clientlevel``. The training-client masks are copied
unchanged; dev/test masks are set to zero because the training code evaluates
dev/test with complete inputs (it does not apply the simulation dictionary to
those clients).

Run from the repository root after ``apply_fixed_masks.py``:

    python features/simulation_features/uci-har/make_clientlevel_files.py --alpha 0.05
"""
import argparse
import json
from pathlib import Path

RATES = (0.5, 0.7, 0.8, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_dir", type=str,
                        default="output/simulation_feature/uci-har")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir)
    alpha_str = str(args.alpha).replace(".", "")

    for rate in RATES:
        tag = f"mm{str(rate).replace('.', '')}"
        src = sim_dir / f"{tag}_alpha{alpha_str}.json"
        if not src.exists():
            raise FileNotFoundError(f"{src} not found; run apply_fixed_masks.py first")

        with open(src) as f:
            data = json.load(f)

        for cid in ("dev", "test"):
            if cid in data:
                for entry in data[cid]:
                    entry[-1][0] = 0
                    entry[-1][1] = 0

        dst = sim_dir / f"{tag}_alpha{alpha_str}_cl.json"
        with open(dst, "w") as f:
            json.dump(data, f)
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
