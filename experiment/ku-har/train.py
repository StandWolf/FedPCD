# ---------------------------------------------------------------------------
# Derived from FedMultimodal (https://github.com/usc-sail/fed-multimodal),
# licensed under the Apache License, Version 2.0.
# Modified for FedPCD (2026). See NOTICE for a summary of changes.
# ---------------------------------------------------------------------------
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.distillation_scheduler import DynamicDistillationScheduler
from utils.prototype_manager import PrototypeManager
from utils.missing_rate import compute_client_missing_rate

import torch
import random
import numpy as np
import torch.nn as nn
import argparse, logging
import copy, pdb
from tqdm import tqdm
from pathlib import Path

from constants import constants
from trainers.server_trainer import Server
from model.mm_models import HARClassifier
from dataloader.dataload_manager import DataloadManager
from trainers.fed_avg_trainer import ClientFedAvg
from trainers.scaffold_trainer import ClientScaffold
from trainers.fed_rs_trainer import ClientFedRS

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

    parser = argparse.ArgumentParser(description='FedPCD KU-HAR experiments')
    parser.add_argument('--data_dir', default=path_conf["output_dir"], type=str, help='output feature directory')
    parser.add_argument('--acc_feat', default='acc', type=str, help="acc feature name")
    parser.add_argument('--gyro_feat', default='gyro', type=str, help="gyro feature name")
    parser.add_argument('--learning_rate', default=0.05, type=float, help="learning rate")
    parser.add_argument('--sample_rate', default=0.1, type=float, help="client sample rate")
    parser.add_argument('--global_learning_rate', default=0.025, type=float, help="global learning rate")
    parser.add_argument('--num_epochs', default=200, type=int, help="total training rounds")
    parser.add_argument('--test_frequency', default=1, type=int, help="perform test frequency")
    parser.add_argument('--local_epochs', default=1, type=int, help="local epochs")
    parser.add_argument('--optimizer', default='sgd', type=str, help="optimizer")
    parser.add_argument('--hid_size', type=int, default=128, help='RNN hidden size dim')
    parser.add_argument('--mu', type=float, default=0.01, help='Fed prox term')
    parser.add_argument('--fed_alg', default='fed_opt', type=str, help="FL aggregation algorithm")
    parser.add_argument('--batch_size', default=16, type=int, help="training batch size")
    parser.add_argument("--missing_modality", type=bool, default=False, help="missing modality simulation")
    parser.add_argument("--en_missing_modality", dest='missing_modality', action='store_true', help="enable missing modality")
    parser.add_argument("--missing_modailty_rate", type=float, default=0.5, help='missing rate for modality')
    parser.add_argument("--missing_label", type=bool, default=False, help="missing label simulation")
    parser.add_argument("--en_missing_label", dest='missing_label', action='store_true', help="enable missing label")
    parser.add_argument("--missing_label_rate", type=float, default=0.5, help='missing rate for label')
    parser.add_argument('--label_nosiy', type=bool, default=False, help='label noise')
    parser.add_argument("--en_label_nosiy", dest='label_nosiy', action='store_true', help="enable label noise")
    parser.add_argument('--label_nosiy_level', type=float, default=0.1, help='nosiy level for labels')
    parser.add_argument('--att', type=bool, default=False, help='self attention')
    parser.add_argument("--en_att", dest='att', action='store_true', help="enable self-attention")
    parser.add_argument('--att_name', type=str, default='fuse_base', help='attention name')
    parser.add_argument("--dataset", type=str, default="ku-har", help='data set name')
    parser.add_argument('--modality', type=str, default='multimodal', help='modality type')

    # Distillation flags
    parser.add_argument('--en_distill', action='store_true', help='enable knowledge distillation')
    parser.add_argument('--distill_alpha_max', default=0.7, type=float, help='max weight for distillation loss')
    parser.add_argument('--distill_init_T', default=5.0, type=float, help='initial temperature')
    parser.add_argument('--distill_min_T', default=1.0, type=float, help='minimum temperature')
    parser.add_argument('--distill_strategy', default='linear', type=str, choices=['linear', 'cosine', 'fixed', 'triangle'])
    parser.add_argument('--distill_gamma', default=0.2, type=float, help='missing-rate coefficient rho')

    # Double distillation flags
    parser.add_argument('--en_double_distill', action='store_true', help='enable double distillation')
    parser.add_argument('--alpha_feat', default=0.2, type=float, help='weight for feature-level distillation')
    parser.add_argument('--lambda_cross', default=0.05, type=float, help='base weight for cross-modal feature distillation')
    parser.add_argument('--cross_eta', default=1.0, type=float, help='enhancement coefficient for cross-modal loss')
    parser.add_argument('--dd_components', default='logit+feat+cross', type=str,
                    help='DD sub-components to enable, separated by +')

    # Proto cross flags
    parser.add_argument('--en_proto_cross', action='store_true', help='enable prototype-based contrastive cross-modal loss')
    parser.add_argument('--proto_ema', default=0.97, type=float, help='EMA momentum for global prototype')
    parser.add_argument('--proto_temperature', default=0.07, type=float, help='temperature for prototype contrastive loss')

    # FedPCD-New leak-free path (B-path single-modality projection) — DEFAULT main path
    parser.add_argument('--leak_free_cross', dest='leak_free_cross', action='store_true', default=True)
    parser.add_argument('--no_leak_free', dest='leak_free_cross', action='store_false')
    parser.add_argument('--leakfree_components', default='rec+proto_paired+proto_missing', type=str)
    parser.add_argument('--lambda_rec', default=1.0, type=float)
    parser.add_argument('--lambda_proto', default=0.05, type=float)
    parser.add_argument('--proto_warmup_start', default=0, type=int)
    parser.add_argument('--proto_warmup_len', default=0, type=int)
    parser.add_argument('--lf_proto_support_ref', default=32, type=int)
    parser.add_argument('--lf_proto_use_alpha', action='store_true')
    parser.add_argument('--missing_protocol', default='clientlevel', type=str,
                    choices=['legacy', 'clientlevel'])

    parser.add_argument('--device', default='cuda:0', type=str, help='device to use')
    parser.add_argument('--num_folds', default=5, type=int, help='number of folds for CV')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device) if args.device.startswith("cuda") else torch.device("cpu")
    print(f'Using device: {device}')

    save_result_dict = dict()

    if args.fed_alg in ['fed_avg', 'fed_prox', 'fed_opt']:
        Client = ClientFedAvg
    elif args.fed_alg in ['scaffold']:
        Client = ClientScaffold
    elif args.fed_alg in ['fed_rs']:
        Client = ClientFedRS

    for fold_idx in range(1, args.num_folds + 1):
        dm = DataloadManager(args)
        dm.get_simulation_setting()
        dm.load_sim_dict(fold_idx=fold_idx)
        dm.get_client_ids(fold_idx=fold_idx)

        dataloader_dict = dict()
        logging.info('Reading Data')
        for client_id in tqdm(dm.client_ids):
            acc_dict = dm.load_acc_feat(fold_idx=fold_idx, client_id=client_id)
            gyro_dict = dm.load_gyro_feat(fold_idx=fold_idx, client_id=client_id)
            dm.get_label_dist(gyro_dict, client_id)
            shuffle = False if client_id in ['dev', 'test'] else True
            client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id=client_id)
            dataloader_dict[client_id] = dm.set_dataloader(
                copy.deepcopy(acc_dict), copy.deepcopy(gyro_dict),
                shuffle=shuffle, client_sim_dict=client_sim_dict,
                default_feat_shape_a=np.array([256, constants.feature_len_dict[args.acc_feat]]),
                default_feat_shape_b=np.array([256, constants.feature_len_dict[args.gyro_feat]]),
            )

        client_ids = [cid for cid in dm.client_ids if cid not in ['dev', 'test']]
        num_of_clients = len(client_ids)

        set_seed(8)
        criterion = nn.NLLLoss().to(device)
        global_model = HARClassifier(
            num_classes=constants.num_class_dict[args.dataset],
            acc_input_dim=constants.feature_len_dict[args.acc_feat],
            gyro_input_dim=constants.feature_len_dict[args.gyro_feat],
            en_att=args.att,
            d_hid=args.hid_size,
            att_name=args.att_name,
        )
        global_model = global_model.to(device)

        server = Server(args, global_model, device=device, criterion=criterion, client_ids=client_ids)
        server.initialize_log(fold_idx)
        server.sample_clients(num_of_clients, sample_rate=args.sample_rate)
        server.get_num_params()

        if args.en_distill:
            distill_scheduler = DynamicDistillationScheduler(
                total_epochs=args.num_epochs,
                init_T=args.distill_init_T, min_T=args.distill_min_T,
                alpha_max=args.distill_alpha_max, strategy=args.distill_strategy,
                gamma=args.distill_gamma
            )
        else:
            distill_scheduler = None

        if args.en_proto_cross:
            prototype_manager = PrototypeManager(
                num_classes=constants.num_class_dict[args.dataset],
                feature_dim=args.hid_size,
                modality_names=["acc_feat", "gyro_feat"],
                device=device,
                ema_momentum=args.proto_ema,
                proto_temperature=args.proto_temperature
            )
            server.prototype_manager = prototype_manager
        else:
            prototype_manager = None

        save_json_path = Path(os.path.realpath(__file__)).parents[2].joinpath(
            'result', args.fed_alg, args.dataset,
            server.feature, server.att, server.model_setting_str
        )
        Path.mkdir(save_json_path, parents=True, exist_ok=True)
        server.save_json_file(dm.label_dist_dict, save_json_path.joinpath('label.json'))

        set_seed(8)
        for epoch in range(int(args.num_epochs)):
            server.initialize_epoch_updates(epoch)
            skip_client_ids = list()
            for idx in server.clients_list[epoch]:
                client_id = client_ids[idx]
                dataloader = dataloader_dict[client_id]

                # Per-client missing rate m_k used by Eq. (19).
                missing_rate = 0.0
                if args.missing_modality:
                    missing_rate = compute_client_missing_rate(
                        dm.get_client_sim_dict(client_id=client_id))

                if dataloader is None:
                    skip_client_ids.append(client_id)
                    continue

                client = Client(
                    args, device, criterion, dataloader,
                    model=copy.deepcopy(server.global_model),
                    label_dict=dm.label_dist_dict[client_id],
                    num_class=constants.num_class_dict[args.dataset],
                    distill_scheduler=distill_scheduler,
                    current_epoch=epoch, missing_rate=missing_rate,
                    prototype_manager=prototype_manager
                )

                if args.fed_alg == 'scaffold':
                    client.set_control(
                        server_control=copy.deepcopy(server.server_control),
                        client_control=copy.deepcopy(server.client_controls[client_id])
                    )
                    client.update_weights()
                    server.set_client_control(client_id, copy.deepcopy(client.client_control))
                    local_protos, local_proto_cnts = client.get_local_prototypes()
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()),
                        client.result['sample'], client.result,
                        delta_control=copy.deepcopy(client.delta_control),
                        local_prototypes=local_protos,
                        local_proto_counts=local_proto_cnts
                    )
                else:
                    client.update_weights()
                    local_protos, local_proto_cnts = client.get_local_prototypes()
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()),
                        client.result['sample'], client.result,
                        local_prototypes=local_protos,
                        local_proto_counts=local_proto_cnts
                    )
                del client

            logging.info(f'Client Round: {epoch}, Skip client {skip_client_ids}')
            if len(server.num_samples_list) == 0: continue
            server.average_weights()
            logging.info('---------------------------------------------------------')
            server.log_classification_result(data_split='train', metric='f1')
            if epoch % args.test_frequency == 0:
                with torch.no_grad():
                    server.inference(dataloader_dict['dev'])
                    server.result_dict[epoch]['dev'] = server.result
                    server.log_classification_result(data_split='dev', metric='f1')
                    server.inference(dataloader_dict['test'])
                    server.result_dict[epoch]['test'] = server.result
                    server.log_classification_result(data_split='test', metric='f1')
                logging.info('---------------------------------------------------------')
                server.log_epoch_result(metric='f1')
            logging.info('---------------------------------------------------------')

        torch.save(
            server.global_model.state_dict(),
            str(save_json_path.joinpath(f'model_final_fold{fold_idx}.pt'))
        )
        save_result_dict[f'fold{fold_idx}'] = server.summarize_dict_results()
        server.save_json_file(save_result_dict, save_json_path.joinpath('result.json'))

    save_result_dict['average'] = dict()
    for metric in ['f1', 'acc', 'top5_acc']:
        result_list = [save_result_dict[k][metric] for k in save_result_dict if metric in save_result_dict[k]]
        save_result_dict['average'][metric] = np.nanmean(result_list)
    server.save_json_file(save_result_dict, save_json_path.joinpath('result.json'))
