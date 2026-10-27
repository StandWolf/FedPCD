# ---------------------------------------------------------------------------
# Derived from FedMultimodal (https://github.com/usc-sail/fed-multimodal),
# licensed under the Apache License, Version 2.0.
# Modified for FedPCD (2026). See NOTICE for a summary of changes.
# ---------------------------------------------------------------------------
import json
import glob
import torch
import pickle
import random
import pdb, os
import numpy as np
import os.path as osp

from PIL import Image
from tqdm import tqdm
from pathlib import Path


class SimulationManager():
    def __init__(self, args: dict):
        self.args = args
        
    def fetch_partition(
        self, 
        fold_idx=1, 
        alpha=0.5,
        ext='json'
    ):
        # reading partition
        if self.args.dataset == "ucf101":
            alpha_str = str(alpha).replace('.', '')
            partition_path = Path(self.args.output_dir).joinpath(
                "partition", 
                self.args.dataset, 
                f'fold{fold_idx}', 
                f'partition_alpha{alpha_str}.{ext}'
            )
        elif self.args.dataset in ["mit10", "mit51", "uci-har", "crisis-mmd", "hateful_memes"]:
            alpha_str = str(alpha).replace('.', '')
            partition_path = Path(self.args.output_dir).joinpath(
                "partition", 
                self.args.dataset, 
                f'partition_alpha{alpha_str}.{ext}'
            )
        elif self.args.dataset in ["meld", "ptb-xl"]:
            partition_path = Path(self.args.output_dir).joinpath(
                "partition", 
                self.args.dataset, 
                f'partition.{ext}'
            )
        elif self.args.dataset in ["crema_d", "ku-har", "extrasensory", "extrasensory_watch"]:
            partition_path = Path(self.args.output_dir).joinpath(
                "partition",
                self.args.dataset,
                f'fold{fold_idx}',
                f'partition.{ext}'
            )
        else:
            raise ValueError(
                f'Dataset not found {self.args.dataset}'
            )

        if ext == "pkl":
            with open(str(partition_path), "rb") as f: 
                partition_dict = pickle.load(f)
        else:
            with open(str(partition_path), "r") as f: 
                partition_dict = json.load(f)

        return partition_dict
    
    def simulate_missing_modality(self, seed):
        np.random.seed(seed)
        keep_prob = 1.0 - self.args.missing_modailty_rate
        mask = np.random.binomial(
            n=1,
            p=keep_prob,
            size=2
        )

        # Ensure at least one modality is available for the client
        if mask.sum() == 0:
            mask[np.random.randint(0, 2)] = 1

        return mask

    def _get_ptbxl_deterministic_mask(self, client_key):
        """Generate deterministic missing modality masks for PTB-XL 33-client setup.

        Distribution per missing rate (33 training clients):
        - 50%: 15 miss I-AVF, 10 miss V1-V6, 8 both present
        - 70%: 12 miss I-AVF, 19 miss V1-V6, 2 both present
        - 80%: 14 miss I-AVF, 18 miss V1-V6, 1 both present
        - 100%: 18 miss I-AVF, 15 miss V1-V6, 0 both present

        No client has both modalities missing simultaneously.
        dev and test clients always have both modalities present.
        """
        if not hasattr(self, '_ptbxl_masks'):
            rate = self.args.missing_modailty_rate
            num_clients = 33

            if rate == 0.5:
                miss_a, miss_b = 10, 15
            elif rate == 0.7:
                miss_a, miss_b = 19, 12
            elif rate == 0.8:
                miss_a, miss_b = 18, 14
            elif rate == 1.0:
                miss_a, miss_b = 15, 18
            else:
                miss_a = int(num_clients * rate * 0.55)
                miss_b = int(num_clients * rate * 0.45)

            masks = []
            # mask[0]=0 I-AVF missing, mask[1]=1 V1-V6 present
            masks.extend([[0, 1]] * miss_a)
            # mask[0]=1 I-AVF present, mask[1]=0 V1-V6 missing
            masks.extend([[1, 0]] * miss_b)
            # mask[0]=1 I-AVF present, mask[1]=1 V1-V6 present
            masks.extend([[1, 1]] * (num_clients - miss_a - miss_b))

            np.random.seed(42)
            np.random.shuffle(masks)
            self._ptbxl_masks = masks

        if client_key.startswith('client_'):
            client_num = int(client_key.split('_')[1])
            if client_num < len(self._ptbxl_masks):
                return np.array(self._ptbxl_masks[client_num])
        return np.array([1, 1])

    def _get_ucihar_deterministic_mask(self, client_idx):
        """Generate deterministic missing modality masks for UCI-HAR 105-client setup.

        Distribution per missing rate (105 training clients):
        - 50%: 40 miss Acc(only Gyro), 35 miss Gyro(only Acc), 30 both present
        - 70%: 55 miss Acc(only Gyro), 43 miss Gyro(only Acc), 7 both present
        - 80%: 48 miss Acc(only Gyro), 54 miss Gyro(only Acc), 3 both present
        - 100%: 50 miss Acc(only Gyro), 55 miss Gyro(only Acc), 0 both present

        mask[0]=0  Acc MISSING, mask[1]=1  Gyro PRESENT → only Gyro
        mask[0]=1  Acc PRESENT, mask[1]=0  Gyro MISSING → only Acc
        mask[0]=1  Acc PRESENT, mask[1]=1  Gyro PRESENT → both
        No client has both modalities missing (mask.sum() > 0 guaranteed).
        """
        if not hasattr(self, '_ucihar_masks'):
            rate = self.args.missing_modailty_rate
            num_clients = 105

            if rate == 0.5:
                miss_a, miss_b = 40, 35   # miss_a=Gyro-only, miss_b=Acc-only
            elif rate == 0.7:
                miss_a, miss_b = 55, 43
            elif rate == 0.8:
                miss_a, miss_b = 48, 54
            elif rate == 1.0:
                miss_a, miss_b = 50, 55
            else:
                miss_a = int(num_clients * rate * 0.53)
                miss_b = int(num_clients * rate * 0.47)

            masks = []
            masks.extend([[0, 1]] * miss_a)   # Acc missing, only Gyro
            masks.extend([[1, 0]] * miss_b)   # Gyro missing, only Acc
            masks.extend([[1, 1]] * (num_clients - miss_a - miss_b))  # both present

            np.random.seed(42)
            np.random.shuffle(masks)
            self._ucihar_masks = masks

        if client_idx < len(self._ucihar_masks):
            return np.array(self._ucihar_masks[client_idx])
        return np.array([1, 1])

    def simulate_missing_label(self, seed, size):
        np.random.seed(seed)
        return np.random.binomial(
            size=size, 
            n=1, 
            p=self.args.missing_label_rate
        )
    
    def label_noise_matrix(
        self, 
        seed, 
        class_num=51
    ):
        # create matrix for each user
        np.random.seed(seed)
        noisy_level = self.args.label_nosiy_level
        sparse_level = 0.4
        prob_matrix = [1-noisy_level] * class_num * class_num
        sparse_elements = np.random.choice(
            class_num*class_num, 
            round(class_num*(class_num-1)*sparse_level)
        )
        for idx in range(len(sparse_elements)):
            while sparse_elements[idx]%(class_num+1) == 0:
                sparse_elements[idx] = np.random.choice(class_num*class_num, 1)
            prob_matrix[sparse_elements[idx]] = 0

        available_spots = np.argwhere(np.array(prob_matrix) == 1 - noisy_level)
        for idx in range(class_num):
            available_spots = np.delete(available_spots, np.argwhere(available_spots == idx*(class_num+1)))

        for idx in range(class_num):
            row = prob_matrix[idx*4:(idx*4)+4]
            if len(np.where(np.array(row) == 1 - noisy_level)[0]) == 2:
                unsafe_points = np.where(np.array(row) == 1 - noisy_level)[0]
                unsafe_points = np.delete(unsafe_points, np.where(np.array(unsafe_points) == idx*(class_num+1))[0])
                available_spots = np.delete(available_spots, np.argwhere(available_spots == unsafe_points[0]))
            if np.sum(row) == 1 - noisy_level:
                zero_spots = np.where(np.array(row) == 0)[0]
                prob_matrix[zero_spots[0] + idx * 4], prob_matrix[available_spots[0]] = prob_matrix[available_spots[0]], prob_matrix[zero_spots[0] + idx * 4]
                available_spots = np.delete(available_spots, 0) 

        prob_matrix = np.reshape(prob_matrix, (class_num, class_num))
        
        for idx in range(len(prob_matrix)):
            zeros = np.count_nonzero(prob_matrix[idx]==0)
            if class_num-zeros-1 == 0:
                prob_element = 0
            else:
                prob_element = (noisy_level) / (class_num-zeros-1)
            prob_matrix[idx] = np.where(prob_matrix[idx] == 1-noisy_level, prob_element, prob_matrix[idx])
            prob_matrix[idx][idx] = 1-noisy_level
            
        return prob_matrix
    
    def get_simulation_setting(self, alpha=None):
        self.setting_str = ''
        if self.args.missing_modality == True:
            self.setting_str += 'mm'+str(self.args.missing_modailty_rate).replace('.', '')
        if self.args.label_nosiy == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ln'+str(self.args.label_nosiy_level).replace('.', '')
        if self.args.missing_label == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ml'+str(self.args.missing_label_rate).replace('.', '')
        if len(self.setting_str) != 0:
            if alpha is not None:
                alpha_str = str(self.args.alpha).replace('.', '')
                self.setting_str += f'_alpha{alpha_str}'
        
    def _get_test_dev_sample_masks(self, num_samples, seed):
        """Generate per-sample missing modality masks for test/dev data.

        Follows the same distribution as training client masks to ensure
        test evaluation matches the paper's protocol where each client's
        test data inherits that client's missing modality status.
        """
        np.random.seed(seed)
        if self.args.dataset == 'uci-har' and hasattr(self, '_ucihar_masks'):
            mask_pool = self._ucihar_masks
        else:
            return np.ones((num_samples, 2), dtype=int)

        # Sample masks from the pool matching train distribution
        indices = np.random.randint(0, len(mask_pool), size=num_samples)
        masks = np.array([mask_pool[i] for i in indices])
        return masks

    def simulation(
        self,
        data_dict: dict,
        seed: int,
        class_num: int=51,
        client_key: str=None
    ) -> (dict):
        is_test_or_dev = client_key in ['test', 'dev']

        # 1. simulate modality missing
        if self.args.missing_modality == True:
            if self.args.dataset == 'ptb-xl' and client_key is not None and not is_test_or_dev:
                mask = self._get_ptbxl_deterministic_mask(client_key)
            elif self.args.dataset == 'uci-har':
                if is_test_or_dev:
                    # For test/dev: trigger mask pool generation, then use per-sample masks
                    self._get_ucihar_deterministic_mask(0)
                    sample_masks = self._get_test_dev_sample_masks(len(data_dict), seed)
                else:
                    mask = self._get_ucihar_deterministic_mask(seed)
                    modality_a_missing = 0 if mask[0] == 1 else 1
                    modality_b_missing = 0 if mask[1] == 1 else 1
            else:
                mask = self.simulate_missing_modality(seed)
                modality_a_missing = 0 if mask[0] == 1 else 1
                modality_b_missing = 0 if mask[1] == 1 else 1
        else:
            modality_a_missing = 0
            modality_b_missing = 0

        # 2. generate label noise matrix for later
        if self.args.label_nosiy == True:
            self.prob_matrix = self.label_noise_matrix(
                seed=seed,
                class_num=class_num
            )

        # 3. missing label simulation
        if self.args.missing_label == True:
            missing_label_array = self.simulate_missing_label(
                seed=seed,
                size=len(data_dict)
            )

        # 2. simulate others
        for idx in range(len(data_dict)):
            # For test/dev with uci-har, use per-sample mask
            if is_test_or_dev and self.args.dataset == 'uci-har' and self.args.missing_modality:
                modality_a_missing = 0 if sample_masks[idx][0] == 1 else 1
                modality_b_missing = 0 if sample_masks[idx][1] == 1 else 1

            # 2.1 simulate label noise
            if self.args.label_nosiy == True:
                orginal_label = data_dict[idx][2]
                np.random.seed(seed)
                if self.args.dataset == 'hateful_memes':
                    change_status = np.random.binomial(size=1, n=1, p=self.args.label_nosiy_level)[0]
                    new_label = orginal_label
                    if change_status == 1:
                        new_label = 1 - orginal_label
                else:
                    new_label = np.random.choice(
                        class_num,
                        p=self.prob_matrix[orginal_label]
                    )
            else:
                new_label = data_dict[idx][2]

            # 2.2 simulate missing label
            if self.args.missing_label == True:
                missing_label = int(missing_label_array[idx])
            else:
                missing_label = 0

            # 2.3 simulation feature vector
            # [missing_modalityA, missing_modalityB, new_label, missing_label]
            data_dict[idx].append([modality_a_missing, modality_b_missing, new_label, missing_label])

        return data_dict