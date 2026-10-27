# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""Evaluate trained models under test-time missingness with per-sample
"same-as-training" masking matching the client-level missing rate m.

For a test set of N samples, each sample is assigned a missing state sampled
(fixed seed) with the same per-modality probabilities as the client-level
training protocol at rate m: P(acc missing)=m/2, P(gyro missing)=m/2,
P(both present)=1-m. At m=1.0 this makes every sample single-modality
(half acc-only, half gyro-only), matching the paper's test-missing convention.

Reports full / acc-only / gyro-only / same-as-training Macro-F1, averaged
over the requested folds. Example:

    python scripts/evaluate_incomplete_inputs.py --dataset uci-har --mm 1.0 \
        --method fedpcd --protocol clientlevel --alpha 0.05 \
        --result_dir result/fed_avg/uci-har/acc_gyro/fuse_base/<run>
"""
import sys, os, copy, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch, numpy as np
import torch.nn as nn
from constants import constants
from model.mm_models import HARClassifier
from model.mfcpl_model import MFCPLModel
from dataloader.dataload_manager import DataloadManager
from trainers.evaluation import EvalMetric

DEVICE = torch.device('cuda:0')
SAMESEED = 2026

class Args:
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)

DS_CONF = {
    'uci-har': dict(dataset='uci-har', sample_rate=0.1, lr=0.05, seq=128, result_ds='uci-har'),
    'mhealth': dict(dataset='mhealth', sample_rate=0.25, lr=0.01, seq=128, result_ds='mhealth_2mod'),
    'ku-har':  dict(dataset='ku-har', sample_rate=0.1, lr=0.05, seq=256, result_ds='ku-har'),
}
METHOD = {
    'fedavg':  ('fed_avg',  False, 'no_att'),
    'fedprox': ('fed_prox', False, 'no_att'),
    'fedopt':  ('fed_opt',  False, 'no_att'),
    'fedpcd':  ('fed_avg',  True,  'fuse_base'),
    'fedmekt': ('fedmekt',  False, 'no_att'),
    'mfcpl':   ('mfcpl',    False, 'no_att'),
}

def load_model(result_dir, dataset, en_att, method, fold=1):
    if method == 'mfcpl':
        m = MFCPLModel(num_classes=constants.num_class_dict[dataset],
                       projection_dim=64, dropout=0.1,
                       in_channels_a=constants.feature_len_dict['acc'],
                       in_channels_b=constants.feature_len_dict['gyro'],
                       hidden_dim=128)
        sd = torch.load(os.path.join(result_dir, f'model_final_fold{fold}.pt'), map_location='cpu')
        m.load_state_dict(sd)
        return m.to(DEVICE)
    # match the training script's att_name so state_dict keys align:
    # train.py (uci) defaults 'multihead'; mhealth/ku FedPCD scripts default 'fuse_base';
    # fedmekt/mfcpl default 'base'.
    if en_att:
        att_name = 'fuse_base'
    elif method == 'fedmekt':
        att_name = 'base'
    elif dataset == 'uci-har':
        att_name = 'multihead'
    else:
        att_name = 'fuse_base'
    m = HARClassifier(num_classes=constants.num_class_dict[dataset],
                      acc_input_dim=constants.feature_len_dict['acc'],
                      gyro_input_dim=constants.feature_len_dict['gyro'],
                      en_att=en_att, d_hid=128, att_name=att_name)
    sd = torch.load(os.path.join(result_dir, f'model_final_fold{fold}.pt'), map_location='cpu')
    m.load_state_dict(sd)
    return m.to(DEVICE)

def find_result_dir(args, method, fold=1):
    if getattr(args, 'result_dir', None):
        return args.result_dir
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    alg, en_att, att = METHOD[method]
    ds = DS_CONF[args.dataset]['result_ds']
    if method == 'fedmekt' and args.dataset == 'mhealth':
        ds = 'mhealth'
    mm = str(args.mm).replace('.', '')
    protocol = getattr(args, 'protocol', 'clientlevel')
    if method == 'mfcpl':
        suffix = '_legacy' if protocol == 'legacy' else ''
        pat = f'{root}/result/{alg}/{ds}/acc_gyro/no_att/*_ep200_mm{mm}*{suffix}'
    else:
        suffix = '_cl' if protocol == 'clientlevel' else ''
        pat = f'{root}/result/{alg}/{ds}/acc_gyro/{att}/*_mm{mm}{suffix}*'
    cands = [d for d in __import__('glob').glob(pat) if '_seed' not in d and '_ep1_' not in d]
    if method == 'fedmekt':
        # legacy fedmekt uses _nokd marker
        if protocol == 'legacy':
            cands = [d for d in cands if '_nokd' in d]
        else:
            cands = [d for d in cands if '_nokd' not in d]
    if method == 'fedpcd':
        cands = [d for d in cands if '_pc_pt007_pe097' in d]
    if protocol == 'legacy':
        cands = [d for d in cands if '_cl' not in d]
    if protocol == 'legacy' and method in ('fedavg', 'fedprox', 'fedopt'):
        # the batch's legacy baselines carry the _lfrecproto naming tag
        cands = [d for d in cands if '_lfrecproto_pairedproto_missing_sr32' in d]
    if method == 'fedopt' and args.dataset == 'uci-har':
        cands = [d for d in cands if '_gl0025' in d]
    for d in cands:
        if os.path.exists(os.path.join(d, f'model_final_fold{fold}.pt')):
            return d
    return None

def make_same_states(n, m):
    """Per-sample [mask_a, mask_b] with P(missing)=m/2 per modality, fixed seed."""
    rng = np.random.RandomState(SAMESEED)
    a = (rng.rand(n) < m / 2).astype(np.float32)
    b = (rng.rand(n) < m / 2).astype(np.float32)
    both = (a == 1) & (b == 1)
    # resolve both-missing: randomly keep one (like client-level at-least-one rule)
    flip = rng.rand(n) < 0.5
    a[both & flip] = 0
    b[both & ~flip] = 0
    return a, b

def run_inference(model, dataloader, mask_a=None, mask_b=None):
    model.eval()
    evaler = EvalMetric(False)
    g = 0
    for batch_data in dataloader:
        x_a, x_b, l_a, l_b, y = batch_data
        x_a = x_a.to(DEVICE); x_b = x_b.to(DEVICE); y = y.to(DEVICE)
        l_a = l_a.to(DEVICE); l_b = l_b.to(DEVICE)
        B = x_a.size(0)
        if mask_a is not None:
            ma = torch.tensor(mask_a[g:g+B], dtype=torch.bool, device=DEVICE)
            mb = torch.tensor(mask_b[g:g+B], dtype=torch.bool, device=DEVICE)
            x_a[ma] = 0; l_a[ma] = 0
            x_b[mb] = 0; l_b[mb] = 0
        g += B
        with torch.no_grad():
            outputs, _, _ = model(x_a.float(), x_b.float(), l_a, l_b)
        outputs = torch.log_softmax(outputs, dim=1)
        evaler.append_classification_results(y, outputs, nn.NLLLoss()(outputs, y))
    return evaler.classification_summary().get('f1', 0)

def eval_fold(args, fold, out_dir, conf):
    a = Args(data_dir=out_dir, dataset=conf['dataset'], acc_feat='acc', gyro_feat='gyro',
             modality='multimodal', hid_size=128, att_name='fuse_base', en_att=True,
             learning_rate=conf['lr'], local_epochs=1, batch_size=16, sample_rate=conf['sample_rate'],
             missing_modality=True, en_missing_modality=True, missing_modailty_rate=args.mm,
             alpha=args.alpha, label_nosiy=False, missing_label=False,
             custom_missing_distribution=None, missing_protocol=args.protocol,
             device=str(DEVICE), num_epochs=1, num_folds=1, test_frequency=1)
    dm = DataloadManager(a)
    dm.get_simulation_setting(alpha=args.alpha if args.dataset == 'uci-har' else None)
    dm.load_sim_dict(fold_idx=fold)
    dm.get_client_ids(fold_idx=fold)
    t_acc = dm.load_acc_feat(fold_idx=fold, client_id='test')
    t_gyro = dm.load_gyro_feat(fold_idx=fold, client_id='test')
    test_dl = dm.set_dataloader(copy.deepcopy(t_acc), copy.deepcopy(t_gyro), shuffle=False,
                                client_sim_dict=None,
                                default_feat_shape_a=np.array([conf['seq'], 3]),
                                default_feat_shape_b=np.array([conf['seq'], 3]))
    n_test = len(t_acc)
    result_dir = find_result_dir(args, args.method, fold)
    if result_dir is None:
        return None
    model = load_model(result_dir, conf['dataset'], METHOD[args.method][1], args.method, fold)
    f1_full = run_inference(model, test_dl)
    f1_acc = run_inference(model, test_dl, mask_a=np.zeros(n_test), mask_b=np.ones(n_test))
    f1_gyro = run_inference(model, test_dl, mask_a=np.ones(n_test), mask_b=np.zeros(n_test))
    ma, mb = make_same_states(n_test, args.mm)
    f1_same = run_inference(model, test_dl, mask_a=ma, mask_b=mb)
    return f1_full, f1_same, f1_acc, f1_gyro

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--mm', type=float, required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--protocol', default='clientlevel', choices=['clientlevel', 'legacy'])
    ap.add_argument('--alpha', type=float, default=0.05,
                    help='Dirichlet alpha used for the UCI-HAR partition (default 0.05)')
    ap.add_argument('--data_dir', default=None,
                    help='directory containing partition/ and simulation_feature/ (default: ./output)')
    ap.add_argument('--result_dir', default=None,
                    help='explicit result directory; skips automatic lookup')
    args = ap.parse_args()
    conf = DS_CONF[args.dataset]
    out_dir = args.data_dir or os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'output'))
    fulls, sames, accs, gyros = [], [], [], []
    for fold in range(1, args.folds + 1):
        r = eval_fold(args, fold, out_dir, conf)
        if r is None:
            print(f"NO MODEL {args.method} {args.dataset} mm{args.mm} fold{fold}"); return
        fulls.append(r[0]); sames.append(r[1]); accs.append(r[2]); gyros.append(r[3])
    print(f"RESULT {args.method:8s} {args.dataset:9s} mm{args.mm}: "
          f"full={np.mean(fulls):.2f} same={np.mean(sames):.2f} acc={np.mean(accs):.2f} gyro={np.mean(gyros):.2f}")

if __name__ == '__main__':
    main()
