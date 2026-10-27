# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""mHealth feature extraction: acc, gyro, mag from right wrist sensor.
Output format per client: [idx, file_ref, label, feature_array]
"""
import pickle, os, argparse, numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-3s ==> %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

def parse_args():
    path_conf = {}
    cfg_path = Path(os.path.realpath(__file__)).parents[3] / 'system.cfg'
    with open(cfg_path) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace('"', '')
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[3] / 'data')
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[3] / 'output')

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', default=path_conf['data_dir'], type=str)
    parser.add_argument('--output_dir', default=path_conf['output_dir'], type=str)
    parser.add_argument('--dataset', default='mhealth', type=str)
    parser.add_argument('--window_size', default=128, type=int)
    return parser.parse_args()

def load_raw_data(raw_dir, subject_id):
    """Load raw log file for a subject, return (acc_rw, gyro_rw, mag_rw) [n_samples, 3]."""
    sid = int(subject_id.split('_')[1])
    fpath = Path(raw_dir) / 'mhealth' / 'MHEALTHDATASET' / f'mHealth_subject{sid}.log'
    data = np.loadtxt(str(fpath), delimiter='\t')
    acc = data[:, 14:17]   # right wrist acc
    gyro = data[:, 17:20]  # right wrist gyro
    mag = data[:, 20:23]   # right wrist mag
    return acc, gyro, mag

def normalize(arr):
    mean, std = np.mean(arr, axis=0), np.std(arr, axis=0)
    return (arr - mean) / (std + 1e-5)

def main():
    args = parse_args()
    raw_dir = args.raw_data_dir
    output_dir = Path(args.output_dir)
    window_size = args.window_size

    # Cache raw data per subject to avoid repeated loading
    raw_cache = {}
    for sid in range(1, 11):
        raw_cache[f'subject_{sid}'] = load_raw_data(raw_dir, f'subject_{sid}')

    for fold_idx in range(1, 6):
        partition_path = output_dir / 'partition' / args.dataset / f'fold{fold_idx}' / 'partition.json'
        import json
        with open(partition_path) as f:
            partition_dict = json.load(f)

        acc_out = output_dir / 'feature' / 'acc' / args.dataset / f'fold{fold_idx}'
        gyro_out = output_dir / 'feature' / 'gyro' / args.dataset / f'fold{fold_idx}'
        mag_out = output_dir / 'feature' / 'mag' / args.dataset / f'fold{fold_idx}'
        for d in [acc_out, gyro_out, mag_out]:
            d.mkdir(parents=True, exist_ok=True)

        for client_id, items in tqdm(partition_dict.items(), desc=f'Fold {fold_idx}'):
            acc_list = []
            gyro_list = []
            mag_list = []

            for item in items:
                idx, file_ref, label = item
                # Parse subject id and window start from file_ref: 'subject_X/window_START'
                subj_str, win_str = file_ref.split('/')
                window_start = int(win_str.split('_')[1])

                acc_raw, gyro_raw, mag_raw = raw_cache[subj_str]
                acc_win = acc_raw[window_start:window_start + window_size]
                gyro_win = gyro_raw[window_start:window_start + window_size]
                mag_win = mag_raw[window_start:window_start + window_size]

                acc_list.append([idx, file_ref, label, normalize(acc_win)])
                gyro_list.append([idx, file_ref, label, normalize(gyro_win)])
                mag_list.append([idx, file_ref, label, normalize(mag_win)])

            if len(acc_list) > 0:
                with open(acc_out / f'{client_id}.pkl', 'wb') as f:
                    pickle.dump(acc_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(gyro_out / f'{client_id}.pkl', 'wb') as f:
                    pickle.dump(gyro_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(mag_out / f'{client_id}.pkl', 'wb') as f:
                    pickle.dump(mag_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info('Feature extraction complete')

if __name__ == '__main__':
    main()
