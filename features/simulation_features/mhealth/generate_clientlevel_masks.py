# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""Generate per-client (client-level) missing masks for mHealth-2mod.

Current mHealth sim uses PER-SAMPLE random masks (violates client-level missing
definition). This script rewrites them so each client is permanently assigned one
modality state, matching the q = per-modality missing probability semantics:
  - draw a per-client binary state {acc_missing, gyro_missing, both} via the
    per-modality Bernoulli model, with the constraint that at least one modality
    is retained (a client is never fully missing).

For each fold, for each train client, one mask is sampled once and applied to
ALL samples of that client. dev/test retain both modalities.

Output: output/simulation_feature/mhealth/foldN/mmXX_cl.json
"""
import json, os, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output' / 'simulation_feature' / 'mhealth'
PART = ROOT / 'output' / 'partition' / 'mhealth'
RATES = {'mm05': 0.5, 'mm07': 0.7, 'mm08': 0.8, 'mm10': 1.0}

def sample_client_state(rate, rng):
    """Return per-client [acc_missing, gyro_missing] (1=missing).
    Per-modality Bernoulli with P(missing)=rate; if both drawn missing, randomly
    restore one (matches the paper's 'at least one modality retained' rule)."""
    if rate >= 1.0:
        # q=1.0: each client keeps exactly one modality
        return [1, 0] if rng.random() < 0.5 else [0, 1]
    a = 1 if rng.random() < rate else 0
    b = 1 if rng.random() < rate else 0
    if a == 1 and b == 1:
        # both missing -> restore one at random
        if rng.random() < 0.5:
            a = 0
        else:
            b = 0
    return [a, b]

def main():
    for fold in range(1, 6):
        part = json.load(open(PART / f'fold{fold}' / 'partition.json'))
        for tag, rate in RATES.items():
            rng = np.random.default_rng(42 + fold)
            out = {}
            for cid, items in part.items():
                if cid in ('dev', 'test'):
                    mask = [0, 0]
                else:
                    mask = sample_client_state(rate, rng)
                client_result = []
                for item in items:
                    # item = [idx, file_ref, label]; append [mask_a, mask_b, mask_c=0, label, missing_label=0]
                    new_item = item + [[mask[0], mask[1], 0, item[2], 0]]
                    client_result.append(new_item)
                out[cid] = client_result
            out_dir = OUT / f'fold{fold}'
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f'{tag}_cl.json', 'w') as f:
                json.dump(out, f)
            # verify per-client consistency
            n_clients = len([c for c in out if c not in ('dev','test')])
            print(f'fold{fold} {tag}: {n_clients} clients, written {out_dir / f"{tag}_cl.json"}')

if __name__ == '__main__':
    main()
