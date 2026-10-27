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
import torchaudio
import numpy as np
import os.path as osp
# import pickle5 as pickle

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, Dataset

def pad_tensor(vec, pad):
    pad_size = list(vec.shape)
    pad_size[0] = pad - vec.size(0)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=0)

def collate_mm_fn_padd(batch):
    # find longest sequence
    if batch[0][0] is not None: max_a_len = max(map(lambda x: x[0].shape[0], batch))
    if batch[0][1] is not None: max_b_len = max(map(lambda x: x[1].shape[0], batch))

    # pad according to max_len
    x_a, x_b, len_a, len_b, ys = list(), list(), list(), list(), list()
    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        x_b.append(pad_tensor(batch[idx][1], pad=max_b_len))
        
        len_a.append(torch.tensor(batch[idx][2]))
        len_b.append(torch.tensor(batch[idx][3]))

        ys.append(batch[idx][-1])
    
    # stack all
    x_a = torch.stack(x_a, dim=0)
    x_b = torch.stack(x_b, dim=0)
    len_a = torch.stack(len_a, dim=0)
    len_b = torch.stack(len_b, dim=0)
    ys = torch.stack(ys, dim=0)
    return x_a, x_b, len_a, len_b, ys

def collate_3modal_fn_padd(batch):
    """Collate 3 modalities: acc, gyro, mag."""
    if batch[0][0] is not None: max_a_len = max(map(lambda x: x[0].shape[0], batch))
    if batch[0][1] is not None: max_b_len = max(map(lambda x: x[1].shape[0], batch))
    if batch[0][2] is not None: max_c_len = max(map(lambda x: x[2].shape[0], batch))

    x_a, x_b, x_c, len_a, len_b, len_c, ys = list(), list(), list(), list(), list(), list(), list()
    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        x_b.append(pad_tensor(batch[idx][1], pad=max_b_len))
        x_c.append(pad_tensor(batch[idx][2], pad=max_c_len))
        len_a.append(torch.tensor(batch[idx][3]))
        len_b.append(torch.tensor(batch[idx][4]))
        len_c.append(torch.tensor(batch[idx][5]))
        ys.append(batch[idx][-1])

    x_a = torch.stack(x_a, dim=0)
    x_b = torch.stack(x_b, dim=0)
    x_c = torch.stack(x_c, dim=0)
    len_a = torch.stack(len_a, dim=0)
    len_b = torch.stack(len_b, dim=0)
    len_c = torch.stack(len_c, dim=0)
    ys = torch.stack(ys, dim=0)
    return x_a, x_b, x_c, len_a, len_b, len_c, ys


def collate_unimodal_fn_padd(batch):
    # find longest sequence
    if batch[0][0] is not None: max_a_len = max(map(lambda x: x[0].shape[0], batch))
    
    # pad according to max_len
    x_a, len_a, ys = list(), list(), list()
    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        len_a.append(torch.tensor(batch[idx][1]))
        ys.append(batch[idx][-1])
    
    # stack all
    x_a = torch.stack(x_a, dim=0)
    len_a = torch.stack(len_a, dim=0)
    ys = torch.stack(ys, dim=0)
    return x_a, len_a, ys


class MMDatasetGenerator(Dataset):
    def __init__(
        self, 
        modalityA, 
        modalityB, 
        default_feat_shape_a,
        default_feat_shape_b,
        data_len: int, 
        simulate_feat=None,
        dataset: str=''
    ):
        
        self.data_len = data_len
        
        self.modalityA = modalityA
        self.modalityB = modalityB
        self.simulate_feat = simulate_feat
        
        self.default_feat_shape_a = default_feat_shape_a
        self.default_feat_shape_b = default_feat_shape_b
        self.dataset = dataset
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        # read modality
        data_a = self.modalityA[item][-1]
        data_b = self.modalityB[item][-1]
        label = torch.tensor(self.modalityA[item][-2])
        # mHealth labels are 1-indexed (1-12), model expects 0-11
        if self.dataset in ['mhealth', 'mhealth_2mod']:
            label = label - 1
        
        # modality A, if missing replace with 0s, and mask
        if data_a is not None: 
            if len(data_a.shape) == 3: data_a = data_a[0]
            data_a = torch.tensor(data_a)
            len_a = len(data_a)
        else: 
            data_a = torch.tensor(np.zeros(self.default_feat_shape_a))
            len_a = 0

        # modality B, if missing replace with 0s
        if data_b is not None:
            if len(data_b.shape) == 3: data_b = data_b[0]
            data_b = torch.tensor(data_b)
            len_b = len(data_b)
        else: 
            data_b = torch.tensor(np.zeros(self.default_feat_shape_b))
            len_b = 0
        return data_a, data_b, len_a, len_b, label



class MM3ModalDatasetGenerator(Dataset):
    """3-modality dataset: acc + gyro + mag."""

    def __init__(self, modalityA, modalityB, modalityC,
                 default_feat_shape_a, default_feat_shape_b, default_feat_shape_c,
                 data_len, simulate_feat=None, dataset=''):
        self.data_len = data_len
        self.modalityA = modalityA
        self.modalityB = modalityB
        self.modalityC = modalityC
        self.simulate_feat = simulate_feat
        self.default_feat_shape_a = default_feat_shape_a
        self.default_feat_shape_b = default_feat_shape_b
        self.default_feat_shape_c = default_feat_shape_c
        self.dataset = dataset

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        data_a = self.modalityA[item][-1]
        data_b = self.modalityB[item][-1]
        data_c = self.modalityC[item][-1]
        label = torch.tensor(self.modalityA[item][-2])
        # mHealth labels are 1-indexed (1-12), model expects 0-11
        if self.dataset in ['mhealth', 'mhealth_2mod']:
            label = label - 1

        if data_a is not None:
            if len(data_a.shape) == 3: data_a = data_a[0]
            data_a = torch.tensor(data_a)
            len_a = len(data_a)
        else:
            data_a = torch.tensor(np.zeros(self.default_feat_shape_a))
            len_a = 0

        if data_b is not None:
            if len(data_b.shape) == 3: data_b = data_b[0]
            data_b = torch.tensor(data_b)
            len_b = len(data_b)
        else:
            data_b = torch.tensor(np.zeros(self.default_feat_shape_b))
            len_b = 0

        if data_c is not None:
            if len(data_c.shape) == 3: data_c = data_c[0]
            data_c = torch.tensor(data_c)
            len_c = len(data_c)
        else:
            data_c = torch.tensor(np.zeros(self.default_feat_shape_c))
            len_c = 0

        return data_a, data_b, data_c, len_a, len_b, len_c, label



class UniModalDatasetGenerator(Dataset):
    def __init__(
        self, 
        modalityA,
        data_len: int, 
        simulate_feat=None,
        dataset: str=''
    ):
        self.dataset = dataset
        self.data_len = data_len
        self.modalityA = modalityA
        self.simulate_feat = simulate_feat
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        # read modality
        data_a = self.modalityA[item][-1]
        label = torch.tensor(self.modalityA[item][-2])
        data_a = torch.tensor(data_a)
        len_a = len(data_a)
        return data_a, len_a, label


class DataloadManager():
    def __init__(
        self, 
        args: dict
    ):
        self.args = args
        self.label_dist_dict = dict()
        # Initialize video feature paths
        if self.args.dataset in ['ucf101', 'mit10', 'mit51', 'mit101', 'crema_d', "ego4d-ttm"]:
            self.get_video_feat_path()
        if self.args.dataset in ['hateful_memes', 'crisis-mmd']:
            self.get_image_feat_path()
            self.get_text_feat_path()
        if self.args.dataset in ['meld']:
            self.get_text_feat_path()
        # Initialize video feature paths for MELD 3-modal
        if self.args.dataset in ['meld'] and getattr(self.args, 'video_feat', None) is not None:
            self.get_video_feat_path()
        # Initialize audio feature paths
        if self.args.dataset in ['ucf101', 'mit10', 'mit51', 'mit101', 'meld', 'crema_d', "ego4d-ttm"]:
            self.get_audio_feat_path()
        # Initialize acc/gyro feature paths
        if self.args.dataset in ['uci-har', 'extrasensory', "ku-har", "mhealth_2mod", "wisdm"]:
            self.get_acc_feat_path()
            self.get_gyro_feat_path()
        # Initialize acc/gyro/mag feature paths
        if self.args.dataset in ['mhealth']:
            self.get_acc_feat_path()
            self.get_gyro_feat_path()
            self.get_mag_feat_path()
        # Initialize acc/watch_acc feature paths
        if self.args.dataset in ['extrasensory_watch']:
            self.get_acc_feat_path()
            self.get_watch_acc_feat_path()
        # Initialize ptb feature paths
        if self.args.dataset in ['ptb-xl']:
            self.i_to_avf_path = Path(self.args.data_dir).joinpath(
                'feature', 
                'I_to_AVF', 
                args.dataset
            )
            self.v1_to_v6_path = Path(self.args.data_dir).joinpath(
                'feature', 
                'V1_to_V6', 
                args.dataset
            )
            
    def get_audio_feat_path(self):
        """
        Load audio feature path.
        """
        self.audio_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'audio', 
            self.args.audio_feat, 
            self.args.dataset
        )
        return Path(self.audio_feat_path)
    
    def get_video_feat_path(self):
        """
        Load frame-wise video feature path.
        """
        self.video_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'video', 
            self.args.video_feat, 
            self.args.dataset
        )
        return Path(self.video_feat_path)
    
    def get_image_feat_path(self):
        """
        Load image feature path.
        """
        self.img_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'img', 
            self.args.img_feat, 
            self.args.dataset
        )
        return Path(self.img_feat_path)

    def get_text_feat_path(self):
        """
        Load text feature path.
        """
        self.text_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'text', 
            self.args.text_feat, 
            self.args.dataset
        )
        return Path(self.text_feat_path)
    
    def get_acc_feat_path(self):
        """
        Load acc feature path.
        """
        self.acc_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'acc', 
            self.args.dataset
        )
        return Path(self.acc_feat_path)
    
    def get_watch_acc_feat_path(self):
        """
        Load watch acc feature path.
        """
        self.watch_acc_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'watch_acc', 
            self.args.dataset
        )
        return Path(self.watch_acc_feat_path)
    
    def get_gyro_feat_path(self):
        """
        Load gyro feature path.
        """
        self.gyro_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 
            'gyro', 
            self.args.dataset
        )
        return Path(self.gyro_feat_path)

    def get_mag_feat_path(self):
        """Load mag feature path."""
        self.mag_feat_path = Path(self.args.data_dir).joinpath(
            'feature', 'mag', self.args.dataset)
        return Path(self.mag_feat_path)

    def get_client_ids(
            self, 
            fold_idx: int=1
        ):
        """
        Load client ids.
        :param fold_idx: fold index
        :return: None
        """
        if self.args.dataset in ["mit10", "mit51"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(
                f'alpha{alpha_str}'
            )
        elif self.args.dataset in ["hateful_memes", "crisis-mmd"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.img_feat_path.joinpath(
                f'alpha{alpha_str}'
            )
        elif self.args.dataset in ["crema_d"]:
            data_path = self.video_feat_path.joinpath(
                f'fold{fold_idx}'
            )
        elif self.args.dataset in ["uci-har"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.acc_feat_path.joinpath(
                f'alpha{alpha_str}'
            )
        elif self.args.dataset in ["wisdm"]:
            data_path = self.acc_feat_path.joinpath('alpha02')
        elif self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(
                f'alpha{alpha_str}', 
                f'fold{fold_idx}'
            )
        elif self.args.dataset == "ego4d-ttm":
            data_path = self.video_feat_path
        elif self.args.dataset in ["extrasensory", "extrasensory_watch", "ku-har", "mhealth", "mhealth_2mod"]:
            data_path = self.acc_feat_path.joinpath(
                f'fold{fold_idx}'
            )
        elif self.args.dataset == "meld":
            data_path = self.text_feat_path
        elif self.args.dataset == "ptb-xl":
            data_path = self.v1_to_v6_path
        self.client_ids = [id.split('.pkl')[0] for id in os.listdir(str(data_path))]
        self.client_ids.sort()
        
    def load_audio_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load audio feature data different applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.audio_feat_path.joinpath(
                f'alpha{alpha_str}', 
                f'fold{fold_idx}', 
                f'{client_id}.pkl'
            )
        elif self.args.dataset == "crema_d":
            data_path = self.audio_feat_path.joinpath(
                f'fold{fold_idx}', 
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["mit10", "mit51"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.audio_feat_path.joinpath(
                f'alpha{alpha_str}', 
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["meld", "ego4d-ttm"]:
            data_path = self.audio_feat_path.joinpath(f'{client_id}.pkl')
        
        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict
    
    def load_video_feat(
            self,
            client_id: str,
            fold_idx: int=1
        ) -> dict:
        """
        Load frame-wise video feature data for MMaction applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "ucf101":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(
                f'alpha{alpha_str}',
                f'fold{fold_idx}',
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["crema_d"]:
            data_path = self.video_feat_path.joinpath(
                f'fold{fold_idx}',
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["ego4d-ttm"]:
            data_path = self.video_feat_path.joinpath(
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["mit10", "mit51"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.video_feat_path.joinpath(
                f'alpha{alpha_str}',
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["meld"]:
            data_path = self.video_feat_path.joinpath(f'{client_id}.pkl')
        with open(str(data_path), "rb") as f:
            data_dict = pickle.load(f)
        return data_dict
    
    def load_img_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load image feature data for image applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset in ["hateful_memes", "crisis-mmd"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.img_feat_path.joinpath(
                f'alpha{alpha_str}', 
                f'{client_id}.pkl'
            )
        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict
    
    def load_acc_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load acc feature data for HAR applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "uci-har":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.acc_feat_path.joinpath(
                f'alpha{alpha_str}',
                f'{client_id}.pkl'
            )
        elif self.args.dataset == "wisdm":
            data_path = self.acc_feat_path.joinpath(
                'alpha02',
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["extrasensory", "extrasensory_watch", 'ku-har', 'mhealth', 'mhealth_2mod']:
            data_path = self.acc_feat_path.joinpath(
                f'fold{fold_idx}',
                f'{client_id}.pkl'
            )
        with open(str(data_path), "rb") as f: data_dict = pickle.load(f)
        return data_dict

    def load_watch_acc_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load watch-based acc feature data for HAR applications.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "extrasensory_watch":
            data_path = self.watch_acc_feat_path.joinpath(
                f'fold{fold_idx}', 
                f'{client_id}.pkl'
            )
        with open(str(data_path), "rb") as f: data_dict = pickle.load(f)
        return data_dict
    
    def load_gyro_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load gyro data for HAR application.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "uci-har":
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.gyro_feat_path.joinpath(
                f'alpha{alpha_str}',
                f'{client_id}.pkl'
            )
        elif self.args.dataset == "wisdm":
            data_path = self.gyro_feat_path.joinpath(
                'alpha02',
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["extrasensory", "ku-har", "mhealth", "mhealth_2mod"]:
            data_path = self.gyro_feat_path.joinpath(
                f'fold{fold_idx}',
                f'{client_id}.pkl'
            )
        with open(str(data_path), "rb") as f: data_dict = pickle.load(f)
        return data_dict

    def load_mag_feat(
            self,
            client_id: str,
            fold_idx: int=1
        ) -> dict:
        """Load mag feature data for HAR applications."""
        if self.args.dataset in ["mhealth", "mhealth_2mod"]:
            data_path = self.mag_feat_path.joinpath(
                f'fold{fold_idx}', f'{client_id}.pkl')
        with open(str(data_path), "rb") as f: data_dict = pickle.load(f)
        return data_dict

    def load_text_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load text feature data.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        if self.args.dataset == "meld":
            data_path = self.text_feat_path.joinpath(
                f'{client_id}.pkl'
            )
        elif self.args.dataset in ["hateful_memes", "crisis-mmd"]:
            alpha_str = str(self.args.alpha).replace('.', '')
            data_path = self.text_feat_path.joinpath(
                f'alpha{alpha_str}', 
                f'{client_id}.pkl'
            )
            
        with open(str(data_path), "rb") as f: 
            data_dict = pickle.load(f)
        return data_dict

    def load_ecg_feat(
            self, 
            client_id: str, 
            fold_idx: int=1
        ) -> dict:
        """
        Load ecg feature data.
        :param client_id: client id
        :param fold_idx: fold index
        :return: data_dict: [key, path, label, feature_array]
        """
        i_to_avf_data_path = self.i_to_avf_path.joinpath(f'{client_id}.pkl')
        v1_to_v6_data_path = self.v1_to_v6_path.joinpath(f'{client_id}.pkl')
        with open(str(i_to_avf_data_path), "rb") as f:  i_to_avf_data_dict = pickle.load(f)
        with open(str(v1_to_v6_data_path), "rb") as f:  v1_to_v6_data_dict = pickle.load(f)
        return i_to_avf_data_dict, v1_to_v6_data_dict

    def get_client_sim_dict(
            self, 
            client_id
        ):
        """
        Set dataloader for training/dev/test.
        :param client_id: client_id
        :return: dataloader: torch dataloader
        """
        if self.sim_data:
            return self.sim_data[client_id]
        return None

    def get_label_dist(
        self, 
        data_dict,
        client_id: str
    ):
        """
        Set dataloader for training/dev/test.
        :param data_dict: data dictionary
        :return: data_dis_dict
        """
        label_list = list()
        for idx in range(len(data_dict)):
            label_list.append(data_dict[idx][-2])
        self.label_dist_dict[client_id] = Counter(label_list)
        
    def set_dataloader(
            self, 
            data_a: dict,
            data_b: dict,
            default_feat_shape_a: np.array=np.array([0, 0]),
            default_feat_shape_b: np.array=np.array([0, 0]),
            client_sim_dict: dict=None,
            shuffle: bool=False
        ) -> (DataLoader):
        """
        Set dataloader for training/dev/test.
        :param data_a: modality A data
        :param data_b: modality B data
        :param default_feat_shape_a: default input shape for modality A, fill 0 in missing modality case
        :param default_feat_shape_b: default input shape for modality B, fill 0 in missing modality case
        :param shuffle: shuffle flag for dataloader, True for training; False for dev and test
        :return: dataloader: torch dataloader
        """
        # modify data based on simulation
        labeled_data_idx, unlabeled_data_idx = list(), list()
        if client_sim_dict is not None:
            for idx in range(len(client_sim_dict)):
                # read simulate feature
                sim_data = client_sim_dict[idx][-1]
                # read modality A
                if sim_data[0] == 1: data_a[idx][-1] = None
                # read modality B
                if sim_data[1] == 1: data_b[idx][-1] = None
                # label: mHealth sim uses 5-element format [mask_a, mask_b, mask_c, label, missing_label]
                # standard sim uses 4-element format [modA_missing, modB_missing, label, missing_label]
                label_idx = 3 if len(sim_data) == 5 else 2
                data_a[idx][-2] = sim_data[label_idx]
                # missing label
                if sim_data[-1] == 0: labeled_data_idx.append(idx)
                else: unlabeled_data_idx.append(idx)

            # return None when ALL samples have both modalities missing
            all_missing = all(
                client_sim_dict[i][-1][0] == 1 and client_sim_dict[i][-1][1] == 1
                for i in range(len(client_sim_dict))
            )
            if all_missing:
                return None

            labeled_data_a, unlabeled_data_a = list(), list()
            labeled_data_b, unlabeled_data_b = list(), list()
            if len(unlabeled_data_idx) > 0:
                for idx in labeled_data_idx:
                    labeled_data_a.append(data_a[idx])
                    labeled_data_b.append(data_b[idx])
                for idx in unlabeled_data_idx:
                    unlabeled_data_a.append(data_a[idx])
                    unlabeled_data_b.append(data_b[idx])
                data_a = labeled_data_a
                data_b = labeled_data_b

        if len(data_a) == 0: return None
        data_ab = MMDatasetGenerator(
            data_a,
            data_b,
            default_feat_shape_a,
            default_feat_shape_b,
            len(data_a),
            self.args.dataset,
            dataset=self.args.dataset
        )
        if shuffle:
            # we use args input batch size for train, typically set as 16 in FL setup
            dataloader = DataLoader(
                data_ab, 
                batch_size=int(self.args.batch_size), 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_mm_fn_padd
            )
        else:
            # we use a larger batch size for validation and testing
            dataloader = DataLoader(
                data_ab, 
                batch_size=64, 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_mm_fn_padd
            )
        return dataloader

    def set_unimodal_dataloader(
            self, 
            data_a: dict,
            client_sim_dict: dict=None,
            shuffle: bool=False
        ) -> (DataLoader):
        """
        Set dataloader for training/dev/test.
        :param data_a: modality A data
        :param shuffle: shuffle flag for dataloader, True for training; False for dev and test
        :return: dataloader: torch dataloader
        """
        # modify data based on simulation
        if client_sim_dict is not None:
            for idx in range(len(client_sim_dict)):
                # read simulate feature
                sim_data = client_sim_dict[idx][-1]
                # pdb.set_trace()
                # read modality A
                if sim_data[0] == 1: data_a[idx][-1] = None
                # read modality B
                if sim_data[1] == 1: data_b[idx][-1] = None
                # label noise
                data_a[idx][-2] = sim_data[2]

            # return None when ALL samples have both modalities missing
            all_missing = all(
                client_sim_dict[i][-1][0] == 1 and client_sim_dict[i][-1][1] == 1
                for i in range(len(client_sim_dict))
            )
            if all_missing:
                return None

        data = UniModalDatasetGenerator(
            data_a, 
            len(data_a),
            self.args.dataset
        )
        if shuffle:
            # we use args input batch size for train, typically set as 16 in FL setup
            dataloader = DataLoader(
                data, 
                batch_size=int(self.args.batch_size), 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_unimodal_fn_padd
            )
        else:
            # we use a larger batch size for validation and testing
            dataloader = DataLoader(
                data, 
                batch_size=64, 
                num_workers=0, 
                shuffle=shuffle, 
                collate_fn=collate_unimodal_fn_padd
            )
        return dataloader

    def set_3modal_dataloader(
            self,
            data_a: dict,
            data_b: dict,
            data_c: dict,
            default_feat_shape_a: np.array = np.array([0, 0]),
            default_feat_shape_b: np.array = np.array([0, 0]),
            default_feat_shape_c: np.array = np.array([0, 0]),
            client_sim_dict: dict = None,
            shuffle: bool = False
        ):
        """Set dataloader for 3-modal training (acc + gyro + mag)."""
        labeled_data_idx, unlabeled_data_idx = list(), list()
        if client_sim_dict is not None:
            for idx in range(len(client_sim_dict)):
                sim_data = client_sim_dict[idx][-1]
                # sim_data format: [mask_a, mask_b, mask_c, label, missing_label]
                # mask=1 means MISSING (set feature to None)
                if int(sim_data[0]) == 1: data_a[idx][-1] = None
                if int(sim_data[1]) == 1: data_b[idx][-1] = None
                if int(sim_data[2]) == 1: data_c[idx][-1] = None
                data_a[idx][-2] = sim_data[3]
                if int(sim_data[4]) == 0:
                    labeled_data_idx.append(idx)
                else:
                    unlabeled_data_idx.append(idx)

            all_missing = all(
                int(client_sim_dict[i][-1][0]) == 1
                and int(client_sim_dict[i][-1][1]) == 1
                and int(client_sim_dict[i][-1][2]) == 1
                for i in range(len(client_sim_dict))
            )
            if all_missing:
                return None

            if len(unlabeled_data_idx) > 0:
                labeled_data_a, unlabeled_data_a = list(), list()
                labeled_data_b, unlabeled_data_b = list(), list()
                labeled_data_c, unlabeled_data_c = list(), list()
                for idx in labeled_data_idx:
                    labeled_data_a.append(data_a[idx])
                    labeled_data_b.append(data_b[idx])
                    labeled_data_c.append(data_c[idx])
                for idx in unlabeled_data_idx:
                    unlabeled_data_a.append(data_a[idx])
                    unlabeled_data_b.append(data_b[idx])
                    unlabeled_data_c.append(data_c[idx])
                data_a, data_b, data_c = labeled_data_a, labeled_data_b, labeled_data_c

        if len(data_a) == 0:
            return None
        data_abc = MM3ModalDatasetGenerator(
            data_a, data_b, data_c,
            default_feat_shape_a, default_feat_shape_b, default_feat_shape_c,
            len(data_a), dataset=self.args.dataset
        )
        batch_size = int(self.args.batch_size) if shuffle else 64
        dataloader = DataLoader(
            data_abc, batch_size=batch_size, num_workers=0,
            shuffle=shuffle, collate_fn=collate_3modal_fn_padd
        )
        return dataloader

    def load_sim_dict(
        self, 
        fold_idx: int=1,
        ext: str="json"
    ):
        """
        Load simulation dictionary.
        :param fold_idx: fold index
        :return: None
        """
        if self.setting_str == '': 
            self.sim_data = None
            return
        
        sim_dataset = self.args.dataset
        if sim_dataset == "mhealth_2mod":
            sim_dataset = "mhealth"  # reuse mhealth simulation features (same clients/folds)
        if sim_dataset in ["ucf101", "crema_d", 'ku-har', 'mhealth', 'extrasensory']:
            data_path = Path(self.args.data_dir).joinpath(
                'simulation_feature',
                sim_dataset,
                f'fold{fold_idx}',
                f'{self.setting_str}.{ext}'
            )
        elif self.args.dataset in ["mit10", "mit51", "meld", "uci-har", "ptb-xl", "crisis-mmd", "hateful_memes", "wisdm"]:
            setting_str = self.setting_str
            if self.args.dataset == 'meld' and getattr(self.args, 'video_feat', None) is not None:
                setting_str = f'{self.setting_str}_3mod'
            data_path = Path(self.args.data_dir).joinpath(
                'simulation_feature',
                self.args.dataset,
                f'{setting_str}.{ext}'
            )
        if ext == "pkl":
            with open(str(data_path), "rb") as f: 
                self.sim_data = pickle.load(f)
        else:
            with open(str(data_path), "r") as f: 
                self.sim_data = json.load(f)
    
    def get_simulation_setting(self, alpha=None):
        """
        Load get simulation setting string.
        :param alpha: alpha in manual split
        :return: None
        """
        self.setting_str = ''
        # 1. missing modality
        if self.args.missing_modality == True:
            self.setting_str += 'mm'+str(self.args.missing_modailty_rate).replace('.', '')
            # Add custom indicator if using custom distribution
            if getattr(self.args, 'custom_missing_distribution', None) is not None:
                self.setting_str += '_custom'
        # 2. label nosiy
        if self.args.label_nosiy == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ln'+str(self.args.label_nosiy_level).replace('.', '')
        # 3. missing labels
        if self.args.missing_label == True:
            if len(self.setting_str) != 0: self.setting_str += '_'
            self.setting_str += 'ml'+str(self.args.missing_label_rate).replace('.', '')
        # 4. alpha for manual split
        if len(self.setting_str) != 0:
            if alpha is not None:
                alpha_str = str(self.args.alpha).replace('.', '')
                self.setting_str += f'_alpha{alpha_str}'
        # 5. client-level missing protocol (Checkpoint 2.5): load mmXX_alpha0X_cl.json
        if (len(self.setting_str) != 0
                and getattr(self.args, 'missing_protocol', 'legacy') == 'clientlevel'):
            self.setting_str += '_cl'