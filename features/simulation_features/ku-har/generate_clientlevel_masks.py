# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""Generate per-client (client-level) missing masks for KU-HAR.

Same protocol as mHealth: each client is permanently assigned one modality state
via per-modality Bernoulli with the constraint that at least one modality is
retained. Mask entry format matches KU-HAR's 4-element layout
[acc_missing, gyro_missing, label, missing_label].

Output: output/simulation_feature/ku-har/foldN/mmXX_cl.json
"""
import json, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output' / 'simulation_feature' / 'ku-har'
PART = ROOT / 'output' / 'partition' / 'ku-har'
RATES = {'mm05': 0.5, 'mm07': 0.7, 'mm08': 0.8, 'mm10': 1.0}

def sample_client_state(rate, rng):
    if rate >= 1.0:
        return [1, 0] if rng.random() < 0.5 else [0, 1]
    a = 1 if rng.random() < rate else 0
    b = 1 if rng.random() < rate else 0
    if a == 1 and b == 1:
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
                mask = [0, 0] if cid in ('dev', 'test') else sample_client_state(rate, rng)
                client_result = []
                for item in items:
                    # item = [idx, file_ref, label] -> append [a, b, label, missing_label=0]
                    new_item = item[:3] + [[mask[0], mask[1], item[2], 0]]
                    client_result.append(new_item)
                out[cid] = client_result
            out_dir = OUT / f'fold{fold}'
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f'{tag}_cl.json', 'w') as f:
                json.dump(out, f)
            n_clients = len([c for c in out if c not in ('dev', 'test')])
            print(f'fold{fold} {tag}: {n_clients} clients')

if __name__ == '__main__':
    main()
