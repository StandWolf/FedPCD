# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""
MFCPL baseline (MFCPL, 2025) on accelerometer + gyroscope HAR datasets.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(root_dir)

import torch, random, numpy as np, torch.nn as nn, argparse, logging, copy
from tqdm import tqdm
from pathlib import Path

from constants import constants
from model.mfcpl_model import MFCPLModel
from dataloader.dataload_manager import DataloadManager
from trainers.mfcpl_server_trainer import MFCPLServer
from trainers.mfcpl_client_trainer import MFCPLClient

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

def set_seed(seed):
    _fast = os.environ.get('FEDPCD_FAST') == '1'
    torch.backends.cudnn.deterministic = not _fast
    torch.backends.cudnn.benchmark = _fast
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[2].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('output'))

    parser = argparse.ArgumentParser(description='MFCPL Training')
    parser.add_argument('--data_dir', default=path_conf["output_dir"], type=str)
    parser.add_argument('--acc_feat', default='acc', type=str)
    parser.add_argument('--gyro_feat', default='gyro', type=str)
    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--sample_rate', default=0.1, type=float)
    parser.add_argument('--global_learning_rate', default=0.025, type=float)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--test_frequency', default=1, type=int)
    parser.add_argument('--local_epochs', default=1, type=int)
    parser.add_argument('--hid_size', type=int, default=128)
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--fed_alg', default='mfcpl', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument("--en_missing_modality", dest='missing_modality', action='store_true')
    parser.add_argument("--missing_modailty_rate", type=float, default=0.5)
    parser.add_argument("--dataset", type=str, default="mhealth", help='mhealth or ku-har')
    parser.add_argument('--modality', type=str, default='multimodal')
    # MFCPL specific
    parser.add_argument('--projection_dim', type=int, default=64)
    parser.add_argument('--alpha_reg', type=float, default=1.0, help='CMPR loss weight')
    parser.add_argument('--alpha_con', type=float, default=2.0, help='CMPC loss weight')
    parser.add_argument('--alpha_align', type=float, default=0.1, help='CMA loss weight')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_folds', default=5, type=int)
    # Additional flags for dataloader compatibility
    parser.add_argument('--label_nosiy', type=bool, default=False)
    parser.add_argument("--en_label_nosiy", dest='label_nosiy', action='store_true')
    parser.add_argument('--label_nosiy_level', type=float, default=0.1)
    parser.add_argument("--missing_label", type=bool, default=False)
    parser.add_argument("--en_missing_label", dest='missing_label', action='store_true')
    parser.add_argument("--missing_label_rate", type=float, default=0.5)
    parser.add_argument('--att', type=bool, default=False)
    parser.add_argument("--en_att", dest='att', action='store_true')
    parser.add_argument('--att_name', type=str, default='base')
    parser.add_argument("--missing_protocol", type=str, default='clientlevel',
                    choices=['legacy', 'clientlevel'],
                    help='legacy or client-level missing protocol')
    parser.add_argument('--alpha', type=float, default=0.2)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device) if args.device.startswith("cuda") else torch.device("cpu")
    print(f'Using device: {device}')

    # Dataset-specific settings
    if args.dataset == 'mhealth':
        args.num_classes = constants.num_class_dict['mhealth']
        seq_len = 128
        sample_rate = 0.25
    elif args.dataset == 'ku-har':
        args.num_classes = constants.num_class_dict['ku-har']
        seq_len = 256
        sample_rate = 0.1
    elif args.dataset == 'uci-har':
        args.num_classes = constants.num_class_dict['uci-har']
        seq_len = 128
        sample_rate = 0.1
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    args.sample_rate = sample_rate
    feat_dim = constants.feature_len_dict[args.acc_feat]  # 3

    save_result_dict = dict()

    for fold_idx in range(1, args.num_folds + 1):
        dm = DataloadManager(args)
        dm.get_simulation_setting(alpha=args.alpha if args.dataset == 'uci-har' else None)
        dm.load_sim_dict(fold_idx=fold_idx)
        dm.get_client_ids(fold_idx=fold_idx)

        dataloader_dict = dict()
        logging.info(f'Reading {args.dataset} Data (Fold {fold_idx})')
        for client_id in tqdm(dm.client_ids):
            acc_dict = dm.load_acc_feat(fold_idx=fold_idx, client_id=client_id)
            gyro_dict = dm.load_gyro_feat(fold_idx=fold_idx, client_id=client_id)
            dm.get_label_dist(gyro_dict, client_id)
            shuffle = False if client_id in ['dev', 'test'] else True
            client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id=client_id)
            dataloader_dict[client_id] = dm.set_dataloader(
                copy.deepcopy(acc_dict), copy.deepcopy(gyro_dict),
                shuffle=shuffle, client_sim_dict=client_sim_dict,
                default_feat_shape_a=np.array([seq_len, feat_dim]),
                default_feat_shape_b=np.array([seq_len, feat_dim]),
            )

        client_ids = [cid for cid in dm.client_ids if cid not in ['dev', 'test']]
        num_of_clients = len(client_ids)

        set_seed(8 * fold_idx)
        criterion = nn.NLLLoss().to(device)
        global_model = MFCPLModel(
            num_classes=args.num_classes,
            projection_dim=args.projection_dim,
            dropout=0.1,
            in_channels_a=feat_dim,
            in_channels_b=feat_dim,
            hidden_dim=args.hid_size,
        )
        global_model = global_model.to(device)

        server = MFCPLServer(
            args=args, model=global_model, device=device,
            criterion=criterion, client_ids=client_ids
        )
        server.initialize_log(fold_idx)
        server.sample_clients(num_of_clients, sample_rate=args.sample_rate)

        # Determine result directory based on dataset
        if args.dataset == 'mhealth':
            dataset_result_dir = 'mhealth_2mod'
        else:
            dataset_result_dir = args.dataset

        model_setting_str = server.model_setting_str
        if getattr(args, 'missing_protocol', 'legacy') != 'clientlevel':
            model_setting_str += '_legacy'
        save_json_path = Path(os.path.realpath(__file__)).parents[2].joinpath(
            'result', args.fed_alg, dataset_result_dir,
            server.feature, server.att, model_setting_str
        )
        Path.mkdir(save_json_path, parents=True, exist_ok=True)

        global_prototypes = server.get_global_prototypes()

        set_seed(8 * fold_idx)
        for epoch in range(int(args.num_epochs)):
            server.initialize_epoch_updates(epoch)
            skip_client_ids = list()
            for idx in server.clients_list[epoch]:
                client_id = client_ids[idx]
                dataloader = dataloader_dict[client_id]
                if dataloader is None:
                    skip_client_ids.append(client_id)
                    continue

                client = MFCPLClient(
                    args=args, device=device, criterion=criterion,
                    dataloader=dataloader,
                    model=copy.deepcopy(global_model),
                    num_class=args.num_classes,
                    global_prototypes=global_prototypes.clone(),
                    alpha_reg=args.alpha_reg,
                    alpha_con=args.alpha_con,
                    alpha_align=args.alpha_align,
                    temperature=args.temperature,
                    client_id=client_id,
                )
                client.update_weights()
                local_prototypes, local_counts = client.get_local_prototypes()
                server.save_train_updates(
                    model_update=client.get_parameters(),
                    num_samples=client.result['sample'],
                    result=client.result,
                    client_prototypes=local_prototypes,
                    client_prototype_counts=local_counts
                )
                del client

            logging.info(f'Client Round: {epoch}, Skip client {skip_client_ids}')
            if len(server.num_samples_list) == 0: continue
            server.aggregate_weights()
            global_prototypes = server.get_global_prototypes().detach()

            if epoch % args.test_frequency == 0:
                server.inference(dataloader_dict['dev'])
                server.result_dict[epoch]['dev'] = server.result
                server.inference(dataloader_dict['test'])
                server.result_dict[epoch]['test'] = server.result
                server.log_classification_result('dev', metric='f1')
                server.log_classification_result('test', metric='f1')
                server.log_epoch_result(metric='f1')
                logging.info('---------------------------------------------------------')

        fold_result = {
            'acc': float(server.best_test_dict['acc']),
            'uar': float(server.best_test_dict['uar']),
            'f1': float(server.best_test_dict['f1'])
        }
        if 'top5_acc' in server.best_test_dict:
            fold_result['top5_acc'] = float(server.best_test_dict['top5_acc'])
        torch.save(
            server.global_model.state_dict(),
            str(save_json_path.joinpath(f'model_final_fold{fold_idx}.pt'))
        )
        save_result_dict[f'fold{fold_idx}'] = fold_result
        server.save_json_file(save_result_dict, save_json_path.joinpath('result.json'))

    save_result_dict['average'] = dict()
    for metric in ['f1', 'acc', 'top5_acc', 'uar']:
        result_list = [save_result_dict[k][metric] for k in save_result_dict if metric in save_result_dict[k] and k != 'average']
        if result_list:
            save_result_dict['average'][metric] = np.nanmean(result_list)
    server.save_json_file(save_result_dict, save_json_path.joinpath('result.json'))

    logging.info(f'\n{"="*60}')
    logging.info(f'MFCPL {args.dataset} Final Results')
    logging.info(f'{"="*60}')
    for k, v in save_result_dict['average'].items():
        logging.info(f'  {k}: {v:.2f}%')
