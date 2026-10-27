# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""
MFCPL Server Trainer
Handles prototype aggregation and distribution for federated learning

Reference: Hong et al., "Multimodal Federated Cross-Prototype Learning", 2024
"""

import torch
import copy
import json
import logging
import numpy as np
from pathlib import Path
# Use tensorboardX instead of torch.utils.tensorboard (more stable on Windows)
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

from .evaluation import EvalMetric


class MFCPLServer:
    """MFCPL Server for prototype-based federated learning"""

    def __init__(self, args, model, device, criterion, client_ids):
        """
        Args:
            args: Configuration arguments
            model: Global model
            device: CUDA/CPU device
            criterion: Loss criterion
            client_ids: List of client IDs
        """
        self.args = args
        self.device = device
        self.global_model = model
        self.criterion = criterion
        self.client_ids = client_ids
        self.multilabel = True if args.dataset == 'ptb-xl' else False

        # MFCPL specific attributes
        self.num_classes = args.num_classes if hasattr(args, 'num_classes') else 6
        self.projection_dim = args.projection_dim if hasattr(args, 'projection_dim') else 64

        # Global prototypes storage
        self.global_prototypes = torch.zeros(
            self.num_classes, self.projection_dim
        ).to(self.device)
        self.prototype_counts = torch.zeros(self.num_classes).to(self.device)

        # Model update tracking
        self.model_updates = []
        self.num_samples_list = []
        self.result_dict = defaultdict(dict)

        # Client prototype updates
        self.client_prototype_updates = {}
        self.client_prototype_counts = {}

        # Logging
        self.model_setting_str = self._get_model_setting()
        self.log_writer = None

    def _get_model_setting(self):
        """Generate model setting string for logging - consistent with Server trainer"""
        model_setting_str = ""
        
        # Dataset-specific feature naming
        if self.args.dataset in ['mit10', 'mit51', 'ucf101', 'crema_d']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.audio_feat}_{self.args.video_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.audio_feat}'
            elif self.args.modality == "video":
                self.feature = f'{self.args.video_feat}'
            model_setting_str += 'alpha'+str(self.args.alpha).replace('.', '')
        elif self.args.dataset in ['hateful_memes', 'crisis-mmd']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.img_feat}_{self.args.text_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.img_feat}'
            elif self.args.modality == "video":
                self.feature = f'{self.args.text_feat}'
            model_setting_str += 'alpha'+str(self.args.alpha).replace('.', '')
        elif self.args.dataset in ['uci-har', 'mhealth', 'wisdm']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.acc_feat}_{self.args.gyro_feat}'
            elif self.args.modality == "acc":
                self.feature = f'{self.args.acc_feat}'
            elif self.args.modality == "gyro":
                self.feature = f'{self.args.gyro_feat}'
            model_setting_str += 'alpha'+str(self.args.alpha).replace('.', '')
        elif self.args.dataset in ['extrasensory', 'ku-har']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.acc_feat}_{self.args.gyro_feat}'
            elif self.args.modality == "acc":
                self.feature = f'{self.args.acc_feat}'
            elif self.args.modality == "gyro":
                self.feature = f'{self.args.gyro_feat}'
        elif self.args.dataset in ['extrasensory_watch']:
            self.feature = f'{self.args.acc_feat}_{self.args.watch_feat}'
        elif self.args.dataset in ['ptb-xl']:
            if self.args.modality == "multimodal":
                self.feature = 'i_to_avf_v1_to_v6'
            elif self.args.modality == "i_to_avf":
                self.feature = 'i_to_avf'
            elif self.args.modality == "v1_to_v6":
                self.feature = 'v1_to_v6'
        elif self.args.dataset in ['meld']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.audio_feat}_{self.args.text_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.audio_feat}'
            elif self.args.modality == "text":
                self.feature = f'{self.args.text_feat}'
        elif self.args.dataset in ['ego4d-ttm']:
            if self.args.modality == "multimodal":
                self.feature = f'{self.args.audio_feat}_{self.args.video_feat}'
            elif self.args.modality == "audio":
                self.feature = f'{self.args.audio_feat}'
            elif self.args.modality == "video":
                self.feature = f'{self.args.video_feat}'
        else:
            raise ValueError(f'Dataset not supported: {self.args.dataset}')
        
        # Training settings
        hid_size = getattr(self.args, 'hid_size', 128)
        if len(model_setting_str) == 0:
            model_setting_str = 'hid'+str(hid_size)
        else:
            model_setting_str += '_hid'+str(hid_size)
        model_setting_str += '_le'+str(self.args.local_epochs)
        model_setting_str += '_lr' + str(self.args.learning_rate).replace('.', '')
        if self.args.fed_alg == 'fed_opt':
            global_lr = getattr(self.args, 'global_learning_rate', 0.025)
            model_setting_str += '_gl'+str(global_lr).replace('.', '')
        model_setting_str += '_bs'+str(self.args.batch_size)
        model_setting_str += '_sr'+str(self.args.sample_rate).replace('.', '')
        model_setting_str += '_ep'+str(self.args.num_epochs)
        if self.args.fed_alg == 'fed_prox':
            mu = getattr(self.args, 'mu', 0.001)
            model_setting_str += '_mu'+str(mu).replace('.', '')
        
        # Attention str
        if getattr(self.args, 'att', False):
            self.att = f'{self.args.att_name}'
        else:
            self.att = 'no_att'

        # FL simulations: missing modality, label noise, missing labels
        if getattr(self.args, 'missing_modality', False):
            mm_rate = getattr(self.args, 'missing_modailty_rate', 
                              getattr(self.args, 'missing_rate', 0.5))
            model_setting_str += '_mm'+str(mm_rate).replace('.', '')
        if getattr(self.args, 'label_nosiy', False):
            model_setting_str += '_ln'+str(self.args.label_nosiy_level).replace('.', '')
        if getattr(self.args, 'missing_label', False):
            model_setting_str += '_ml'+str(self.args.missing_label_rate).replace('.', '')
        
        # Knowledge distillation
        if hasattr(self.args, 'en_distill') and self.args.en_distill:
            model_setting_str += '_da'+str(self.args.distill_alpha_max).replace('.', '')
        
        # Feature decoupling
        if hasattr(self.args, 'en_decouple') and self.args.en_decouple:
            model_setting_str += '_dc'  # decouple
        
        # Cross-modal generation
        if hasattr(self.args, 'en_cross_modal_gen') and self.args.en_cross_modal_gen:
            model_setting_str += '_gm' + str(self.args.gamma_gen_max).replace('.', '')
        
        # MFCPL specific parameters
        if self.args.fed_alg == 'mfcpl':
            alpha_reg = getattr(self.args, 'alpha_reg', 1.0)
            alpha_con = getattr(self.args, 'alpha_con', 2.0)
            alpha_align = getattr(self.args, 'alpha_align', 0.1)
            temperature = getattr(self.args, 'temperature', 0.1)
            model_setting_str += f'_ar{alpha_reg:.1f}_ac{alpha_con:.1f}_aa{alpha_align:.1f}_t{temperature:.1f}'.replace('.', '')
        
        return model_setting_str

    def initialize_log(self, fold_idx=1):
        """Initialize logging for current fold - consistent with Server trainer"""
        self.fold_idx = fold_idx

        # Log path - now includes feature and att levels
        self.log_path = Path(self.args.data_dir).joinpath(
            'log', 
            self.args.fed_alg,
            self.args.dataset,
            self.feature,
            self.att,
            self.model_setting_str,
            f'fold{fold_idx}', 
            'raw_log'
        )
        self.result_path = Path(__file__).parents[2].joinpath(
            'result',
            self.args.fed_alg,
            self.args.dataset,
            self.feature,
            self.att,
            self.model_setting_str
        )
        
        # Convert to absolute path and create directories
        self.log_path = self.log_path.resolve()
        self.result_path = self.result_path.resolve()
        
        # Create directories with explicit mode
        import os
        import time
        os.makedirs(str(self.log_path), exist_ok=True)
        os.makedirs(str(self.result_path), exist_ok=True)
        
        # Verify directories exist and are writable
        if not self.log_path.exists():
            raise RuntimeError(f"Failed to create log directory: {self.log_path}")
        if not self.result_path.exists():
            raise RuntimeError(f"Failed to create result directory: {self.result_path}")
        
        # Test write access
        test_file = self.log_path / '.write_test'
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            test_file.unlink()
        except Exception as e:
            logging.warning(f"Directory write test failed: {e}")
        
        logging.info(f'Log path: {self.log_path}')
        logging.info(f'Result path: {self.result_path}')

        # Windows fix: Try creating SummaryWriter with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.log_writer = SummaryWriter(
                    log_dir=str(self.log_path)
                )
                # Force initialization with a write
                self.log_writer.add_scalar('init', 0, 0)
                self.log_writer.flush()
                break
            except FileNotFoundError as e:
                if attempt < max_retries - 1:
                    logging.warning(f"SummaryWriter creation failed (attempt {attempt+1}), retrying...")
                    time.sleep(0.5)
                    # Recreate directory
                    os.makedirs(str(self.log_path), exist_ok=True)
                else:
                    logging.error(f"Failed to create SummaryWriter after {max_retries} attempts: {e}")
                    raise

        self.best_test_dict = None
        self.best_epoch = 0

    def sample_clients(self, num_of_clients, sample_rate=0.1):
        """Sample clients for each communication round"""
        import numpy as np

        self.clients_list = []
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
        """Initialize updates for current epoch"""
        self.epoch = epoch
        self.model_updates = []
        self.num_samples_list = []
        self.client_prototype_updates = {}
        self.client_prototype_counts = {}
        self.result_dict[self.epoch] = {
            'train': [],
            'dev': [],
            'test': {}
        }

    def get_parameters(self):
        """Get current global model parameters"""
        return self.global_model.state_dict()

    def aggregate_weights(self):
        """
        Aggregate model weights using simple average (mean aggregation)
        Also aggregate client prototypes
        """
        if len(self.num_samples_list) == 0:
            return

        num_clients = len(self.model_updates)

        # Aggregate model weights (simple average)
        w_avg = copy.deepcopy(self.model_updates[0])

        for key in w_avg.keys():
            w_avg[key] = self.model_updates[0][key] / num_clients

        for key in w_avg.keys():
            for i in range(1, len(self.model_updates)):
                w_avg[key] += self.model_updates[i][key] / num_clients

        # Apply FedOpt optimization if needed
        if self.args.fed_alg == 'fed_opt':
            self._update_global_with_optimizer(w_avg)
        else:
            self.global_model.load_state_dict(w_avg)

        # Aggregate prototypes
        self._aggregate_prototypes()

    def _update_global_with_optimizer(self, update_weights):
        """Update global model using server-side optimizer (FedOpt)"""
        global_optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            lr=self.args.global_learning_rate,
            momentum=0.9,
            weight_decay=0.0
        )

        # Zero gradients
        global_optimizer.zero_grad()

        # Create new model with aggregated weights
        new_model = copy.deepcopy(self.global_model)
        new_model.load_state_dict(update_weights, strict=True)

        # Compute pseudo-gradients
        with torch.no_grad():
            for param, new_param in zip(
                self.global_model.parameters(),
                new_model.parameters()
            ):
                param.grad = (param.data - new_param.data) / self.args.learning_rate

        # Optimization step
        global_optimizer.step()

    def _aggregate_prototypes(self):
        """
        Aggregate client prototypes to update global prototypes
        Formula: P_k = (1/N) * sum_{clients} p_client_k  (simple average)
        """
        if not self.client_prototype_updates:
            return

        # Initialize with zeros
        aggregated_prototypes = torch.zeros(
            self.num_classes, self.projection_dim
        ).to(self.device)
        client_count = len(self.client_prototype_updates)

        # Aggregate all client prototypes (simple average)
        for client_id in self.client_prototype_updates:
            client_proto = self.client_prototype_updates[client_id]

            # Simple average across clients
            for k in range(self.num_classes):
                aggregated_prototypes[k] += client_proto[k]

        # Normalize by number of clients
        for k in range(self.num_classes):
            if client_count > 0:
                self.global_prototypes[k] = aggregated_prototypes[k] / client_count
            else:
                # Reset to zeros if no clients
                self.global_prototypes[k] = torch.zeros(self.projection_dim).to(self.device)

        # Store counts for next round (set to client count for each class)
        self.prototype_counts = torch.full((self.num_classes,), client_count).to(self.device)

    def get_global_prototypes(self):
        """Get current global prototypes"""
        return self.global_prototypes.clone()

    def save_train_updates(self, model_update, num_samples, result,
                          client_prototypes=None, client_prototype_counts=None):
        """
        Save client training updates
        Args:
            model_update: Client model state dict
            num_samples: Number of samples used
            result: Client training results
            client_prototypes: Client prototypes (dict or tensor)
            client_prototype_counts: Sample counts per class
        """
        self.model_updates.append(model_update)
        self.num_samples_list.append(num_samples)
        self.result_dict[self.epoch]['train'].append(result)

        # Save prototype updates
        if client_prototypes is not None and client_prototype_counts is not None:
            # Convert dict to tensor if needed
            if isinstance(client_prototypes, dict):
                proto_tensor = torch.stack([
                    client_prototypes[k] for k in range(self.num_classes)
                ]).to(self.device)
            else:
                proto_tensor = client_prototypes.to(self.device)

            if isinstance(client_prototype_counts, dict):
                count_tensor = torch.tensor([
                    client_prototype_counts[k] for k in range(self.num_classes)
                ]).to(self.device)
            else:
                count_tensor = client_prototype_counts.to(self.device)

            # Store (use a simple counter as client_id)
            client_id = len(self.client_prototype_updates)
            self.client_prototype_updates[client_id] = proto_tensor
            self.client_prototype_counts[client_id] = count_tensor

    def inference(self, dataloader):
        """Run inference on dataloader"""
        self.global_model.eval()

        eval_metric = EvalMetric(self.multilabel)

        with torch.no_grad():
            for batch_data in dataloader:
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)

                    outputs, _, _ = self.global_model(x_a.float(), x_b.float(), l_a, l_b)
                else:
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)

                    outputs, _, _ = self.global_model(x.float(), l)

                # Compute loss
                if not self.multilabel:
                    outputs = torch.log_softmax(outputs, dim=1)
                loss = self.criterion(outputs, y)

                # Save results
                if not self.multilabel:
                    eval_metric.append_classification_results(y, outputs, loss)
                else:
                    eval_metric.append_multilabel_results(y, outputs, loss)

        # Get summary
        if not self.multilabel:
            self.result = eval_metric.classification_summary()
        else:
            self.result = eval_metric.multilabel_summary()

    def log_classification_result(self, data_split, metric='acc'):
        """Log classification results (supports both single-label and multilabel)"""
        if data_split == 'train':
            loss = np.mean([d['loss'] for d in self.result_dict[self.epoch][data_split]])
            acc = np.mean([d['acc'] for d in self.result_dict[self.epoch][data_split]])
            if not self.multilabel:
                top5_acc = np.mean([d['top5_acc'] for d in self.result_dict[self.epoch][data_split]])
                f1 = np.mean([d['f1'] for d in self.result_dict[self.epoch][data_split]])
            else:
                macro_f = np.mean([d['macro_f'] for d in self.result_dict[self.epoch][data_split]])
        else:
            result = self.result_dict[self.epoch][data_split]
            loss = result['loss']
            acc = result['acc']
            if not self.multilabel:
                top5_acc = result['top5_acc']
                f1 = result['f1']
            else:
                macro_f = result['macro_f']

        # Console logging
        if data_split == 'train':
            logging.info(f'Current Round: {self.epoch}')

        if self.multilabel:
            logging.info(f'{data_split} set, Loss: {loss:.3f}, Macro-F1: {macro_f:.2f}%, Acc: {acc:.2f}%')
        elif metric == 'acc':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, Acc: {acc:.2f}%, Top-5 Acc: {top5_acc:.2f}%')
        elif metric == 'f1':
            logging.info(f'{data_split} set, Loss: {loss:.3f}, Macro-F1: {f1:.2f}%, Top-1 Acc: {acc:.2f}%')

        # TensorBoard logging
        if data_split != 'train':
            self.log_writer.add_scalar(f'Loss/{data_split}', loss, self.epoch)
            self.log_writer.add_scalar(f'Acc/{data_split}', acc, self.epoch)
            if not self.multilabel:
                self.log_writer.add_scalar(f'Top5_Acc/{data_split}', top5_acc, self.epoch)
                self.log_writer.add_scalar(f'F1/{data_split}', f1, self.epoch)
            else:
                self.log_writer.add_scalar(f'Macro_F1/{data_split}', macro_f, self.epoch)

    def log_epoch_result(self, metric='acc'):
        """Log best results for current epoch"""
        dev_result = self.result_dict[self.epoch]['dev']
        test_result = self.result_dict[self.epoch]['test']

        if self.best_test_dict is None:
            self.best_epoch = self.epoch
            self.best_dev_dict = dev_result
            self.best_test_dict = test_result

            # Save best model
            torch.save(
                copy.deepcopy(self.global_model.state_dict()),
                str(self.result_path.joinpath('model.pt'))
            )

        # Update if dev improves
        if dev_result[metric] > self.best_dev_dict[metric]:
            self.best_epoch = self.epoch
            self.best_dev_dict = dev_result
            self.best_test_dict = test_result

            # Save best model
            torch.save(
                copy.deepcopy(self.global_model.state_dict()),
                str(self.result_path.joinpath('model.pt'))
            )

        # Log best results
        logging.info(f'Best epoch {self.best_epoch}')
        if self.multilabel:
            logging.info(f'Best dev Acc: {self.best_dev_dict["acc"]:.2f}%, Macro-F1: {self.best_dev_dict["macro_f"]:.2f}%')
            logging.info(f'Best test Acc: {self.best_test_dict["acc"]:.2f}%, Macro-F1: {self.best_test_dict["macro_f"]:.2f}%')
        else:
            logging.info(f'Best dev Acc: {self.best_dev_dict["acc"]:.2f}%, F1: {self.best_dev_dict["f1"]:.2f}%')
            logging.info(f'Best test Acc: {self.best_test_dict["acc"]:.2f}%, F1: {self.best_test_dict["f1"]:.2f}%')

    def save_result(self, file_path):
        """Save results to JSON file"""
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return obj

        # Convert result_dict to JSON-serializable format
        serializable_dict = json.loads(
            json.dumps(self.result_dict, default=convert_to_serializable)
        )

        with open(str(file_path), 'w', encoding='utf-8') as f:
            json.dump(serializable_dict, f, indent=4, ensure_ascii=False)

    def save_prototypes(self, file_path):
        """Save global prototypes to file"""
        torch.save({
            'global_prototypes': self.global_prototypes.cpu(),
            'prototype_counts': self.prototype_counts.cpu()
        }, str(file_path))

    def summarize_results(self):
        """Summarize final results"""
        import pandas as pd

        row_df = pd.DataFrame(index=[f'fold{self.fold_idx}'])
        row_df['acc'] = self.best_test_dict['acc']
        if not self.multilabel:
            row_df['top5_acc'] = self.best_test_dict['top5_acc']
            row_df['uar'] = self.best_test_dict['uar']
            row_df['f1'] = self.best_test_dict['f1']
        else:
            row_df['macro_f'] = self.best_test_dict['macro_f']
        return row_df

    def summarize_dict_results(self):
        """Summarize results as dictionary - consistent with Server trainer"""
        result = dict()
        result['acc'] = self.best_test_dict['acc']
        if not self.multilabel:
            result['top5_acc'] = self.best_test_dict['top5_acc']
            result['uar'] = self.best_test_dict['uar']
            result['f1'] = self.best_test_dict['f1']
            if "auc" in self.best_test_dict:
                result['auc'] = self.best_test_dict['auc']
        else:
            result['macro_f'] = self.best_test_dict['macro_f']
        return result

    def save_json_file(self, data_dict, data_path):
        """Save data to JSON file - consistent with Server trainer"""
        import json
        jsonString = json.dumps(data_dict, indent=4)
        jsonFile = open(str(data_path), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    def get_num_params(self):
        """Get number of model parameters"""
        import numpy as np
        model_parameters = filter(lambda p: p.requires_grad, self.global_model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters]) / 1000
        logging.info(f'Number of Parameters: {num_params} K')
        return num_params
