# ---------------------------------------------------------------------------
# Derived from FedMultimodal (https://github.com/usc-sail/fed-multimodal),
# licensed under the Apache License, Version 2.0.
# Modified for FedPCD (2026). See NOTICE for a summary of changes.
# ---------------------------------------------------------------------------
import json
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch

from pathlib import Path
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from .evaluation import EvalMetric

# logging format
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


class Server(object):
    def __init__(
        self,
        args,
        model,
        device,
        criterion,
        client_ids,
        prototype_manager=None
    ):
        self.args = args
        self.device = device
        self.result_dict = dict()
        self.global_model = model
        self.criterion = criterion
        self.client_ids = client_ids
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        self.model_setting_str = self.get_model_setting()
        self.prototype_manager = prototype_manager

        if self.args.fed_alg == 'scaffold':
            self.server_control = self.init_control(model)
            self.set_control_device(self.server_control, True)
            self.client_controls = {
                client_id: self.init_control(model) for client_id in self.client_ids
            }
        elif self.args.fed_alg == 'fed_opt':
            self.global_optimizer = self._initialize_global_optimizer()
            
    def _initialize_global_optimizer(self):
        # global optimizer
        global_optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            lr=self.args.global_learning_rate,
            momentum=0.9,
            weight_decay=0.0
        )
        return global_optimizer

    def set_client_control(
        self,
        client_id,
        client_control
    ):
        self.client_controls[client_id] = client_control
        
    def initialize_log(
        self, 
        fold_idx: int=1
    ):
        # log saving path
        self.fold_idx = fold_idx
        # Use shorter log path to avoid Windows 260 char limit
        self.log_path = Path('output/log').joinpath(
            f'{self.args.fed_alg}_{self.args.dataset}_{fold_idx}'
        )
        self.result_path = Path('output/log').joinpath(
            f'{self.args.fed_alg}_{self.args.dataset}_{fold_idx}'
        )
        Path.mkdir(self.log_path, parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(
            str(self.log_path), 
            filename_suffix=f'_{self.model_setting_str}'
        )
        
        self.best_test_dict = list()
        
    def get_model_setting(self):
        # Return model setting
        model_setting_str = ""
        # Audio video pairs
        if self.args.dataset in ['mit10', 'mit51', 'ucf101', 'crema_d']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.audio_feat}_{self.args.video_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.audio_feat}'
            elif self.args.modality == "video":
                self.feature = f'{self.args.video_feat}'
            model_setting_str += 'alpha'+str(self.args.alpha).replace('.', '')
        # Image text pairs
        elif self.args.dataset in ['hateful_memes', 'crisis-mmd']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.img_feat}_{self.args.text_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.img_feat}'
            elif self.args.modality == "video":
                self.feature = f'{self.args.text_feat}'
            model_setting_str += 'alpha'+str(self.args.alpha).replace('.', '')
        # Acc gyro pairs
        elif self.args.dataset in ['uci-har', 'wisdm']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.acc_feat}_{self.args.gyro_feat}'
            elif self.args.modality == "acc":
                self.feature = f'{self.args.acc_feat}'
            elif self.args.modality == "gyro":
                self.feature = f'{self.args.gyro_feat}'
            model_setting_str += 'alpha'+str(getattr(self.args, 'alpha', 0.2)).replace('.', '')
        elif self.args.dataset in ['extrasensory', 'ku-har']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.acc_feat}_{self.args.gyro_feat}'
            elif self.args.modality == "acc":
                self.feature = f'{self.args.acc_feat}'
            elif self.args.modality == "gyro":
                self.feature = f'{self.args.gyro_feat}'
        elif self.args.dataset in ['mhealth', 'mhealth_2mod']:
            if self.args.modality == "multimodal":
                if hasattr(self.args, 'mag_feat'):
                    self.feature = f'{self.args.acc_feat}_{self.args.gyro_feat}_{self.args.mag_feat}'
                else:
                    self.feature = f'{self.args.acc_feat}_{self.args.gyro_feat}'
            elif self.args.modality == "acc":
                self.feature = f'{self.args.acc_feat}'
            elif self.args.modality == "gyro":
                self.feature = f'{self.args.gyro_feat}'
            elif self.args.modality == "mag":
                self.feature = f'{self.args.mag_feat}'
        elif self.args.dataset in ['extrasensory_watch']:
            model_setting_str = f'{self.args.acc_feat}_{self.args.watch_feat}'
        elif self.args.dataset in ['ptb-xl']:
            if self.args.modality == "multimodal":
                self.feature = 'i_to_avf_v1_to_v6'
            elif self.args.modality == "i_to_avf":
                self.feature = 'i_to_avf'
            elif self.args.modality == "v1_to_v6":
                self.feature = 'v1_to_v6'
        # Audio text pairs
        elif self.args.dataset in ['meld']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.audio_feat}_{self.args.text_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.audio_feat}'
            elif self.args.modality == "text":
                self.feature = f'{self.args.text_feat}'
        # Ego4Ds
        elif self.args.dataset in ['ego4d-ttm']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.audio_feat}_{self.args.video_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.audio_feat}'
            elif self.args.modality == "video":
                self.feature = f'{self.args.video_feat}'
        else:
            raise ValueError(f'Data set not support {self.args.dataset}')
        # training settings: local epochs, learning rate, batch size, client sample rate
        if len(model_setting_str) == 0:
            model_setting_str = 'hid'+str(self.args.hid_size)
        else:
            model_setting_str += '_hid'+str(self.args.hid_size)
        model_setting_str += '_le'+str(self.args.local_epochs)
        model_setting_str += '_lr' + str(self.args.learning_rate).replace('.', '')
        if self.args.fed_alg == 'fed_opt': model_setting_str += '_gl'+str(self.args.global_learning_rate).replace('.', '')
        model_setting_str += '_bs'+str(self.args.batch_size)
        model_setting_str += '_sr'+str(self.args.sample_rate).replace('.', '')
        model_setting_str += '_ep'+str(self.args.num_epochs)
        if self.args.fed_alg == 'fed_prox': model_setting_str += '_mu'+str(self.args.mu).replace('.', '')
        
        # attention str
        if self.args.att: self.att = f'{self.args.att_name}'
        else: self.att = 'no_att'

        # FL simulations: missing modality, label noise, missing labels
        if self.args.missing_modality == True:
            model_setting_str += '_mm'+str(self.args.missing_modailty_rate).replace('.', '')
            # Add custom indicator if using custom distribution
            if getattr(self.args, 'custom_missing_distribution', None) is not None:
                model_setting_str += '_custom'
            # client-level missing protocol (Checkpoint 2.5) -> distinct result dir vs legacy
            if getattr(self.args, 'missing_protocol', 'legacy') == 'clientlevel':
                model_setting_str += '_cl'
        # FedPCD-New leak-free variant tag -> distinct result dir per variant
        if getattr(self.args, 'leak_free_cross', False):
            _lf = getattr(self.args, 'leakfree_components', 'rec+proto_paired+proto_missing').replace('+', '')
            model_setting_str += '_lf' + _lf
            # encode support_reference so S5 (sr1) and S6 (sr32) don't collide
            _sr = getattr(self.args, 'lf_proto_support_ref', 32)
            model_setting_str += f'_sr{_sr}'
        # Training seed for result isolation
        _ts = getattr(self.args, 'train_seed', 0)
        if _ts != 0:
            model_setting_str += f'_seed{_ts}'
        if self.args.label_nosiy == True:
            model_setting_str += '_ln'+str(self.args.label_nosiy_level).replace('.', '')
        if self.args.missing_label == True:
            model_setting_str += '_ml'+str(self.args.missing_label_rate).replace('.', '')
        # Knowledge distillation
        if hasattr(self.args, 'en_distill') and self.args.en_distill:
            model_setting_str += '_da'+str(self.args.distill_alpha_max).replace('.', '')
            if getattr(self.args, 'distill_strategy', 'linear') != 'linear':
                model_setting_str += '_ds' + self.args.distill_strategy
        # Extensions
        if getattr(self.args, 'en_double_distill', False):
            model_setting_str += '_dd'
            # Encode cross-modal loss parameters to differentiate ablations
            if hasattr(self.args, 'lambda_cross'):
                model_setting_str += '_lc' + str(self.args.lambda_cross).replace('.', '')
            if hasattr(self.args, 'cross_eta'):
                model_setting_str += '_ce' + str(self.args.cross_eta).replace('.', '')
            if hasattr(self.args, 'alpha_feat'):
                model_setting_str += '_af' + str(self.args.alpha_feat).replace('.', '')
            # Encode DD sub-component selection for ablation studies
            dd_comp = getattr(self.args, 'dd_components', 'logit+feat+cross')
            if dd_comp != 'logit+feat+cross':
                model_setting_str += '_' + dd_comp.replace('+', '-')
        if getattr(self.args, 'en_proto_cross', False):
            model_setting_str += '_pc'
            if hasattr(self.args, 'proto_temperature'):
                model_setting_str += '_pt' + str(self.args.proto_temperature).replace('.', '')
            if hasattr(self.args, 'proto_ema'):
                model_setting_str += '_pe' + str(self.args.proto_ema).replace('.', '')
        if getattr(self.args, 'en_distill', False) and self.args.dataset == 'ptb-xl':
            model_setting_str += '_kl'
        if getattr(self.args, 'no_federated_proj', False):
            model_setting_str += '_nofp'
        return model_setting_str
    
    def sample_clients(self, num_of_clients, sample_rate=0.1):
        # Sample clients per round
        self.clients_list = list()
        for epoch in range(int(self.args.num_epochs)):
            np.random.seed(epoch)
            idxs_clients = np.random.choice(
                range(num_of_clients), 
                int(sample_rate * num_of_clients), 
                replace=False
            )
            self.clients_list.append(idxs_clients)
        self.num_of_clients = num_of_clients

    def initialize_epoch_updates(self, epoch):
        # Initialize updates
        self.epoch = epoch
        self.model_updates = list()
        self.num_samples_list = list()
        self.delta_controls = list()
        self.local_proto_list = list()
        self.local_proto_count_list = list()
        self.result_dict[self.epoch] = dict()
        self.result_dict[self.epoch]['train'] = list()
        self.result_dict[self.epoch]['dev'] = list()
        self.result_dict[self.epoch]['test'] = list()
        
    def get_parameters(self):
        # Return model parameters
        return self.global_model.state_dict()
    
    def get_model_result(self):
        # Return model results
        return self.result
    
    def get_test_true(self):
        # Return test labels
        return self.test_true
    
    def get_test_pred(self):
        # Return test predictions
        return self.test_pred
    
    def get_train_groundtruth(self):
        # Return groundtruth used for training
        return self.train_groundtruth
    
    def update(self, learning_rate, label_list, fed_setting):
        if fed_setting == 'fed_avg': self.update_weights(learning_rate, label_list)
        else: self.update_gradients(learning_rate, label_list)

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.global_model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters]) / 1000
        logging.info(f'Number of Parameters: {num_params} K')
        return num_params
    
    def inference(self, dataloader):
        self.global_model.eval()

        # initialize eval
        self.eval = EvalMetric(self.multilabel)
        for batch_idx, batch_data in enumerate(dataloader):
                
            self.global_model.zero_grad()
            if self.args.modality == "multimodal":
                if self.args.dataset == "mhealth" and hasattr(self.args, 'mag_feat'):
                    x_a, x_b, x_c, l_a, l_b, l_c, y = batch_data
                    x_a = x_a.to(self.device); x_b = x_b.to(self.device); x_c = x_c.to(self.device)
                    y = y.to(self.device)
                    l_a = l_a.to(self.device); l_b = l_b.to(self.device); l_c = l_c.to(self.device)
                    outputs, _, feats = self.global_model(
                        x_a.float(), x_b.float(), x_c.float(), l_a, l_b, l_c)
                elif self.args.dataset == "meld" and hasattr(self.args, 'video_feat') and self.args.video_feat is not None:
                    x_a, x_b, x_c, l_a, l_b, l_c, y = batch_data
                    x_a = x_a.to(self.device); x_b = x_b.to(self.device); x_c = x_c.to(self.device)
                    y = y.to(self.device)
                    l_a = l_a.to(self.device); l_b = l_b.to(self.device); l_c = l_c.to(self.device)
                    outputs, _, feats = self.global_model(
                        x_a.float(), x_b.float(), l_a, l_b, x_c.float(), l_c)
                else:
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                    outputs, _, feats = self.global_model(
                        x_a.float(), x_b.float(), l_a, l_b)
            else:
                x, l, y = batch_data
                x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                
                # forward
                outputs, _, feats = self.global_model(
                    x.float(), l
                )
        
            if not self.multilabel: 
                outputs = torch.log_softmax(outputs, dim=1)
            loss = self.criterion(outputs, y)
            
            # save results
            if not self.multilabel: 
                self.eval.append_classification_results(
                    y, outputs, loss
                )
            else:
                self.eval.append_multilabel_results(
                    y, outputs, loss
                )
                
        # epoch train results
        if not self.multilabel:
            if self.args.dataset in ["hateful_memes", "ego4d-ttm"]:
                self.result = self.eval.classification_summary(return_auc=True)
            else:
                self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()

    def log_classification_result(
            self, 
            data_split: str, 
            metric: str='uar'
        ):
        if data_split == 'train':
            loss = np.mean([data['loss'] for data in self.result_dict[self.epoch][data_split]])
            acc = np.mean([data['acc'] for data in self.result_dict[self.epoch][data_split]])
            uar = np.mean([data['uar'] for data in self.result_dict[self.epoch][data_split]])
            top5_acc = np.mean([data['top5_acc'] for data in self.result_dict[self.epoch][data_split]])
            f1 = np.mean([data['f1'] for data in self.result_dict[self.epoch][data_split]])
        else:
            loss = self.result_dict[self.epoch][data_split]['loss']
            acc = self.result_dict[self.epoch][data_split]['acc']
            uar = self.result_dict[self.epoch][data_split]['uar']
            top5_acc = self.result_dict[self.epoch][data_split]['top5_acc']
            f1 = self.result_dict[self.epoch][data_split]['f1']
            if metric == 'auc':
                auc = self.result_dict[self.epoch][data_split]['auc']

        # loggin console
        if data_split == 'train': logging.info(f'Current Round: {self.epoch}')
        if metric == 'acc':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, Acc: {acc:.2f}%, Top-5 Acc: {top5_acc:.2f}%')
        elif metric == 'f1':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, Macro-F1: {f1:.2f}%, Top-1 Acc: {acc:.2f}%')
        elif metric == 'uar':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, UAR: {uar:.2f}%, Top-1 Acc: {acc:.2f}%')
        
        if metric == 'auc' and data_split != 'train':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, AUC: {auc:.2f}%, Top-1 Acc: {acc:.2f}%')
        # loggin to folder
        self.log_writer.add_scalar(f'Loss/{data_split}', loss, self.epoch)
        self.log_writer.add_scalar(f'Acc/{data_split}', acc, self.epoch)
        self.log_writer.add_scalar(f'UAR/{data_split}', uar, self.epoch)
        self.log_writer.add_scalar(f'F1/{data_split}', f1, self.epoch)
        self.log_writer.add_scalar(f'Top5_Acc/{data_split}', top5_acc, self.epoch)
        if metric == 'auc' and data_split != 'train': self.log_writer.add_scalar(f'AUC/{data_split}', auc, self.epoch)
        
    def log_multilabel_result(
        self, 
        data_split: str, 
        metric: str='macro_f'
    ):
        if data_split == 'train':
            loss = np.mean([data['loss'] for data in self.result_dict[self.epoch][data_split]])
            acc = np.mean([data['acc'] for data in self.result_dict[self.epoch][data_split]])
            macro_f = np.mean([data['macro_f'] for data in self.result_dict[self.epoch][data_split]])
        else:
            loss = self.result_dict[self.epoch][data_split]['loss']
            acc = self.result_dict[self.epoch][data_split]['acc']
            macro_f = self.result_dict[self.epoch][data_split]['macro_f']

        # logging to console
        if data_split == 'train':
            # 记录各损失分量
            loss_task_avg = np.mean([d.get('loss_task', 0) for d in self.result_dict[self.epoch][data_split]])
            loss_logit_avg = np.mean([d.get('loss_logit', 0) for d in self.result_dict[self.epoch][data_split]])
            loss_feat_com_avg = np.mean([d.get('loss_feat_common', 0) for d in self.result_dict[self.epoch][data_split]])
            loss_feat_cross_avg = np.mean([d.get('loss_feat_cross', 0) for d in self.result_dict[self.epoch][data_split]])
            logging.info(
                f'Current Round: {self.epoch}'
            )
            logging.info(
                f'{data_split} set, Loss: {loss:.3f}, Macro-F1: {macro_f:.2f}%, Top-1 Acc: {acc:.2f}%'
            )
            logging.info(
                f'  Loss components: task={loss_task_avg:.3f}, logit={loss_logit_avg:.3f}, '
                f'feat_common={loss_feat_com_avg:.3f}, feat_cross={loss_feat_cross_avg:.3f}'
            )
        else:
            logging.info(
                f'{data_split} set, Loss: {loss:.3f}, Macro-F1: {macro_f:.2f}%, Top-1 Acc: {acc:.2f}%'
            )

        # logging to folder
        self.log_writer.add_scalar(f'Loss/{data_split}', loss, self.epoch)
        self.log_writer.add_scalar(f'Acc/{data_split}', acc, self.epoch)
        self.log_writer.add_scalar(f'Macro-F1/{data_split}', macro_f, self.epoch)
        if data_split == 'train':
            self.log_writer.add_scalar('LossComp/task', loss_task_avg, self.epoch)
            self.log_writer.add_scalar('LossComp/logit', loss_logit_avg, self.epoch)
            self.log_writer.add_scalar('LossComp/feat_common', loss_feat_com_avg, self.epoch)
            self.log_writer.add_scalar('LossComp/feat_cross', loss_feat_cross_avg, self.epoch)

    def save_result(
        self, 
        file_path
    ):
        jsonString = json.dumps(
            self.result_dict, 
            indent=4
        )
        jsonFile = open(str(file_path), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    def save_train_updates(
        self,
        model_updates: dict,
        num_sample: int,
        result: dict,
        delta_control=None,
        local_prototypes=None,
        local_proto_counts=None
    ):
        self.model_updates.append(model_updates)
        self.num_samples_list.append(num_sample)
        self.result_dict[self.epoch]['train'].append(result)
        self.delta_controls.append(delta_control)
        if local_prototypes is not None:
            self.local_proto_list.append(local_prototypes)
            self.local_proto_count_list.append(local_proto_counts)

    def log_epoch_result(
        self, 
        metric: str='acc'
    ):
        if len(self.best_test_dict) == 0:
            self.best_epoch = self.epoch
            self.best_dev_dict = self.result_dict[self.epoch]['dev']
            self.best_test_dict = self.result_dict[self.epoch]['test']

        if self.result_dict[self.epoch]['dev'][metric] > self.best_dev_dict[metric]:
            # Save best model and training history
            self.best_epoch = self.epoch
            self.best_dev_dict = self.result_dict[self.epoch]['dev']
            self.best_test_dict = self.result_dict[self.epoch]['test']
            torch.save(
                deepcopy(self.global_model.state_dict()), 
                str(self.result_path.joinpath('model.pt'))
            )
        
        # log dev results
        best_dev_acc = self.best_dev_dict['acc']
        if not self.multilabel:
            best_dev_uar = self.best_dev_dict['uar']
            best_dev_top5_acc = self.best_dev_dict['top5_acc']
            best_dev_macro_f1 = self.best_dev_dict['f1']
            if metric == "auc": best_dev_auc = self.best_dev_dict['auc']
        else:
            best_dev_macro_f1 = self.best_dev_dict['macro_f']

        # log test results
        best_test_acc = self.best_test_dict['acc']
        if not self.multilabel:
            best_test_uar = self.best_test_dict['uar']
            best_test_macro_f1 = self.best_test_dict['f1']
            best_test_top5_acc = self.best_test_dict['top5_acc']
            if metric == "auc": best_test_auc = self.best_test_dict['auc']
        else:
            best_test_macro_f1 = self.best_test_dict['macro_f']
            
        # logging
        logging.info(f'Best epoch {self.best_epoch}')
        if metric == 'acc':
            logging.info(f'Best dev Top-1 Acc {best_dev_acc:.2f}%, Top-5 Acc {best_dev_top5_acc:.2f}%')
            logging.info(f'Best test Top-1 Acc {best_test_acc:.2f}%, Top-5 Acc {best_test_top5_acc:.2f}%')
        elif metric == 'macro_f':
            logging.info(f'Best dev Macro-F1 {best_dev_macro_f1:.2f}%, Top-1 Acc {best_dev_acc:.2f}%')
            logging.info(f'Best test Macro-F1 {best_test_macro_f1:.2f}%, Top-1 Acc {best_test_acc:.2f}%')
        elif metric == 'f1':
            logging.info(f'Best dev Macro-F1 {best_dev_macro_f1:.2f}%, Top-1 Acc {best_dev_acc:.2f}%')
            logging.info(f'Best test Macro-F1 {best_test_macro_f1:.2f}%, Top-1 Acc {best_test_acc:.2f}%')
        elif metric == 'auc':
            logging.info(f'Best dev AUC {best_dev_auc:.2f}%, Top-1 Acc {best_dev_acc:.2f}%')
            logging.info(f'Best test AUC {best_test_auc:.2f}%, Top-1 Acc {best_test_acc:.2f}%')
        else:
            logging.info(f'Best dev UAR {best_dev_uar:.2f}%, Top-1 Acc {best_dev_acc:.2f}%')
            logging.info(f'Best test UAR {best_test_uar:.2f}%, Top-1 Acc {best_test_acc:.2f}%')

    def summarize_results(self):
        row_df = pd.DataFrame(index=[f'fold{self.fold_idx}'])
        row_df['acc']  = self.best_test_dict['acc']
        if not self.multilabel:
            row_df['top5_acc'] = self.best_test_dict['top5_acc']
            row_df['uar'] = self.best_test_dict['uar']
            if "auc" in self.best_test_dict: row_df['auc'] = self.best_test_dict['auc']
        else:
            row_df['macro_f'] = self.best_test_dict['macro_f']
        return row_df

    def summarize_dict_results(self):
        result = dict()
        if isinstance(self.best_test_dict, list):
            result['acc'] = float('nan')
            result['f1'] = float('nan')
            result['auc'] = float('nan')
            if not self.multilabel:
                result['top5_acc'] = float('nan')
                result['uar'] = float('nan')
            else:
                result['macro_f'] = float('nan')
            return result
        result['acc'] = self.best_test_dict['acc']
        if not self.multilabel:
            result['top5_acc'] = self.best_test_dict['top5_acc']
            result['uar'] = self.best_test_dict['uar']
            result['f1'] = self.best_test_dict['f1']
            if "auc" in self.best_test_dict: result['auc'] = self.best_test_dict['auc']
        else:
            result['macro_f'] = self.best_test_dict['macro_f']
        return result

    def average_weights(self):
        """
        Returns the average of the weights.
        """
        # there are no samples, return
        if len(self.num_samples_list) == 0: 
            return

        # 标准 FedAvg 聚合
        total_num_samples = np.sum(self.num_samples_list)
        w_avg = copy.deepcopy(self.model_updates[0])

        # calculate weighted updates
        for key in w_avg.keys():
            if self.args.fed_alg == 'scaffold':
                w_avg[key] = torch.div(self.model_updates[0][key], len(self.model_updates))
            else:
                w_avg[key] = self.model_updates[0][key]*(self.num_samples_list[0]/total_num_samples)
        for key in w_avg.keys():
            for i in range(1, len(self.model_updates)):
                if self.args.fed_alg == 'scaffold':
                    w_avg[key] += torch.div(self.model_updates[i][key], len(self.model_updates))
                else:
                    w_avg[key] += torch.div(self.model_updates[i][key]*self.num_samples_list[i], total_num_samples)
        
        # server optimization or just load with weights
        if self.args.fed_alg == 'fed_opt':
            self.update_global(copy.deepcopy(w_avg))
        else:
            strict = not getattr(self.args, 'no_federated_proj', False)
            self.global_model.load_state_dict(copy.deepcopy(w_avg), strict=strict)

        # aggregate prototypes if enabled
        if self.prototype_manager is not None and len(self.local_proto_list) > 0:
            self._aggregate_prototypes()

        # update global control if algorithm is scaffold
        if self.args.fed_alg == 'scaffold':
            self.update_server_control()

    def _aggregate_prototypes(self):
        """聚合客户端上传的局部原型（加权平均后 EMA 更新）。"""
        n_updated = 0
        for i, (lp, lc) in enumerate(zip(self.local_proto_list, self.local_proto_count_list)):
            if lp is not None and lc is not None:
                self.prototype_manager.update(lp, lc)
                n_updated += 1

        # Diagnostic: prototype quality (every 10 epochs)
        if self.epoch % 10 == 0 and n_updated > 0:
            proto_stats = self.prototype_manager.get_stats()
            if proto_stats:
                import logging
                parts = [f'[Diag-Proto] E{self.epoch}']
                for mod_name, stats in proto_stats.items():
                    parts.append(
                        f'{mod_name}:active={stats["active"]}/{stats["total"]},'
                        f'|p|={stats["mean_norm"]:.2f},'
                        f'cos_var={stats["cosine_variance"]:.4f}'
                    )
                logging.info(' '.join(parts))

    def update_server_control(self):
        # update server control
        total_num_samples = np.sum(self.num_samples_list)
        delta_avg = copy.deepcopy(self.delta_controls[0])
        new_control = copy.deepcopy(self.server_control)

        for key in delta_avg.keys():
            delta_avg[key] = torch.div(self.delta_controls[0][key], self.num_of_clients)
            
        for key in delta_avg.keys():
            for idx in range(1, len(self.delta_controls)):
                delta_avg[key] += torch.div(self.delta_controls[idx][key], self.num_of_clients)
            new_control[key] = new_control[key] - delta_avg[key]
        self.server_control = copy.deepcopy(new_control)
        
    def update_global(
        self, 
        update_weights
    ):
        # zero_grad
        self.global_optimizer.zero_grad()
        global_optimizer_state = self.global_optimizer.state_dict()

        # new_model
        new_model = copy.deepcopy(self.global_model)
        new_model.load_state_dict(update_weights, strict=True)

        # set global_model gradient
        with torch.no_grad():
            for param, new_param in zip(
                self.global_model.parameters(), new_model.parameters()
            ):
                param.grad = (param.data - new_param.data) / self.args.learning_rate

        # replace some non-parameters's state dict
        state_dict = self.global_model.state_dict()
        for name in dict(self.global_model.named_parameters()).keys():
            update_weights[name] = state_dict[name]
        self.global_model.load_state_dict(update_weights, strict=True)

        # optimization
        self.global_optimizer = self._initialize_global_optimizer()
        self.global_optimizer.load_state_dict(global_optimizer_state)
        self.global_optimizer.step()

    def set_save_json_file(
        self, 
        file_path
    ):
        self.save_json_path = file_path


    def save_json_file(
        self,
        data_dict,
        data_path
    ):
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def walk(o):
            if isinstance(o, dict):
                return {k: walk(v) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return [walk(v) for v in o]
            return convert(o)

        jsonString = json.dumps(
            walk(data_dict),
            indent=4
        )
        jsonFile = open(str(data_path), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    # scaffold init contral
    def init_control(self, model):
        """ a dict type: {name: params}
        """
        control = {
            name: torch.zeros_like(
                p.data
            ).cpu() for name, p in model.state_dict().items()
        }
        return control

    def set_control_device(
        self, 
        control, 
        device=True
    ):
        for name in control.keys():
            if device == True:
                control[name] = control[name].to(self.device)
            else:
                control[name] = control[name].cpu()
