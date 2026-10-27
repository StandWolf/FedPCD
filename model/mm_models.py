#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Derived from FedMultimodal (https://github.com/usc-sail/fed-multimodal),
# licensed under the Apache License, Version 2.0.
# Modified for FedPCD (2026). See NOTICE for a summary of changes.
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@author: Tiantian
"""
import pdb
import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

# typing import
from typing import Dict, Iterable, Optional


class MMActionClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        audio_input_dim: int,   # Audio feature input dim
        video_input_dim: int,   # Frame-wise video feature input dim
        d_hid: int=128,         # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False,     # Enable self attention or not
        att_name: str='',       # Attention Name
        d_head: int=6           # Head dim
    ):
        super(MMActionClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        
        # Conv Encoder module
        self.audio_conv = Conv1dEncoder(
            input_dim=audio_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
        self.audio_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid,
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        self.video_rnn = nn.GRU(
            input_size=video_input_dim, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )
        
        # Attention modules
        if self.att_name == "multihead":
            self.audio_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
            
            self.video_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
        elif self.att_name == "additive":
            self.audio_att = AdditiveAttention(
                d_hid=d_hid, 
                d_att=128
            )
            self.video_att = AdditiveAttention(
                d_hid=d_hid, 
                d_att=128
            )
        elif self.att_name == "base":
            self.audio_att = BaseSelfAttention(
                d_hid=d_hid
            )
            self.video_att = BaseSelfAttention(
                d_hid=d_hid
            )
        elif self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )
        elif self.att_name == "hirarchical":
            self.att = HirarchicalAttention(
                d_hid=rnn_input
            )
        
        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            # Projection head
            self.audio_proj = nn.Linear(d_hid, d_hid//2)
            self.video_proj = nn.Linear(d_hid, d_hid//2)
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
            
         # Projection head
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(
        self, 
        x_audio, 
        x_video, 
        len_a, 
        len_v
    ):
        # 1. Conv forward
        x_audio = self.audio_conv(x_audio)
        
        # 2. Rnn forward
        # max pooling, time dim reduce by 8 times
        len_a = len_a//8
        if len_a[0] != 0:
            x_audio = pack_padded_sequence(
                x_audio, 
                len_a.cpu().numpy(), 
                batch_first=True, 
                enforce_sorted=False
            )
        if len_v[0] != 0:
            x_video = pack_padded_sequence(
                x_video, 
                len_v.cpu().numpy(), 
                batch_first=True, 
                enforce_sorted=False
            )

        x_audio, _ = self.audio_rnn(x_audio) 
        x_video, _ = self.video_rnn(x_video) 
        if len_a[0] != 0:
            x_audio, _ = pad_packed_sequence(   
                x_audio, 
                batch_first=True
            )
        if len_v[0] != 0:
            x_video, _ = pad_packed_sequence(
                x_video, 
                batch_first=True
            )

        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_video, _ = self.video_att(x_video, x_video, x_video)
                # 4. Average pooling
                x_audio = torch.mean(x_audio, axis=1)
                x_video = torch.mean(x_video, axis=1)
            elif self.att_name == 'additive':
                # get attention output
                x_audio = self.audio_att(x_audio, x_audio, x_audio, len_a)
                x_video = self.video_att(x_video, x_video, x_video, len_v)
            elif self.att_name == "fuse_base":
                # get attention output
                a_max_len = x_audio.shape[1]
                x_mm = torch.cat((x_audio, x_video), dim=1)
                x_mm = self.fuse_att(x_mm, len_a, len_v, a_max_len)
            elif self.att_name == 'base':
                # get attention output
                x_audio = self.audio_att(x_audio)
                x_video = self.video_att(x_video, len_v)
        else:
            # 4. Average pooling
            x_audio = torch.mean(x_audio, axis=1)
            x_video = torch.mean(x_video, axis=1)
            x_mm = torch.cat((x_audio, x_video), dim=1)

        # 5. Projection with no attention
        if self.en_att and self.att_name != "fuse_base":
            x_audio = self.audio_proj(x_audio)
            x_video = self.video_proj(x_video)
            x_mm = torch.cat((x_audio, x_video), dim=1)
        # 6. MM embedding and predict
        preds = self.classifier(x_mm)

        # 特征字典（用于双蒸馏等扩展）
        feat_dict = {"audio_feat": x_audio if x_audio.dim() == 2 else torch.mean(x_audio, axis=1),
                     "video_feat": x_video if x_video.dim() == 2 else torch.mean(x_video, axis=1)}
        return preds, x_mm, feat_dict


class SERClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,       # Number of classes
        audio_input_dim: int,   # Audio data input dim
        text_input_dim: int,    # Text data input dim
        d_hid: int=64,          # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False,     # Enable self attention or not
        att_name: str='',       # Attention Name
        d_head: int=6,          # Head dim
        video_input_dim: int=None,  # Video data input dim (None = 2-modal)
    ):
        super(SERClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.num_modalities = 3 if video_input_dim is not None else 2

        # Conv Encoder module
        self.audio_conv = Conv1dEncoder(
            input_dim=audio_input_dim,
            n_filters=n_filters,
            dropout=self.dropout_p,
        )

        # RNN module
        self.audio_rnn = nn.GRU(
            input_size=n_filters*4,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False
        )

        self.text_rnn = nn.GRU(
            input_size=text_input_dim,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False
        )

        # Video RNN (no conv encoder — MobileNetV2 features are already extracted)
        if video_input_dim is not None:
            self.video_rnn = nn.GRU(
                input_size=video_input_dim,
                hidden_size=d_hid,
                num_layers=1,
                batch_first=True,
                dropout=self.dropout_p,
                bidirectional=False
            )

        # Self attention module
        if self.att_name == "multihead":
            self.audio_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid,
                num_heads=4,
                dropout=self.dropout_p
            )
            self.text_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid,
                num_heads=4,
                dropout=self.dropout_p
            )
            if video_input_dim is not None:
                self.video_att = torch.nn.MultiheadAttention(
                    embed_dim=d_hid,
                    num_heads=4,
                    dropout=self.dropout_p
                )
        elif self.att_name == "base":
            self.audio_att = BaseSelfAttention(
                d_hid=d_hid
            )
            self.text_att = BaseSelfAttention(
                d_hid=d_hid
            )
            if video_input_dim is not None:
                self.video_att = BaseSelfAttention(d_hid=d_hid)
        elif self.att_name == "fuse_base":
            if video_input_dim is not None:
                self.fuse_att = FuseBaseSelfAttention3Mod(
                    d_hid=d_hid,
                    d_head=d_head
                )
            else:
                self.fuse_att = FuseBaseSelfAttention(
                    d_hid=d_hid,
                    d_head=d_head
                )

        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
            cross_input_dim = d_hid * d_head
        else:
            # Projection head
            self.audio_proj = nn.Linear(d_hid, d_hid//2)
            self.text_proj = nn.Linear(d_hid, d_hid//2)
            if video_input_dim is not None:
                self.video_proj = nn.Linear(d_hid, d_hid//2)
            self.init_weight()

            # classifier head
            # With projection: d_hid → d_hid//2 per modality → total = d_hid//2 * N
            # Without projection (no_att): d_hid per modality → total = d_hid * N
            if en_att:
                classifier_input_dim = (d_hid // 2) * self.num_modalities
            else:
                classifier_input_dim = d_hid * self.num_modalities
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
            cross_input_dim = classifier_input_dim

        # Cross-modal projection heads for FedPCD
        self.cross_modal_adapt = nn.Sequential(
            nn.Linear(cross_input_dim, d_hid),
            nn.ReLU()
        )
        cross_heads = {
            "audio_feat": nn.Sequential(
                nn.Linear(d_hid, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, d_hid)
            ),
            "text_feat": nn.Sequential(
                nn.Linear(d_hid, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, d_hid)
            )
        }
        if video_input_dim is not None:
            cross_heads["video_feat"] = nn.Sequential(
                nn.Linear(d_hid, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, d_hid)
            )
        self.cross_modal_heads = nn.ModuleDict(cross_heads)

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_audio, x_text, len_a, len_t, x_video=None, len_v=None):
        # 1. Conv forward
        x_audio = self.audio_conv(x_audio)

        # 2. Rnn forward
        # max pooling, time dim reduce by 8 times
        len_a = len_a//8
        len_a[len_a==0] = 1
        if len_a[0] != 0:
            x_audio = pack_padded_sequence(
                x_audio,
                len_a.cpu().numpy(),
                batch_first=True,
                enforce_sorted=False
            )
        len_t[len_t == 0] = 1
        if len_t[0] != 0:
            x_text = pack_padded_sequence(
                x_text,
                len_t.cpu().numpy(),
                batch_first=True,
                enforce_sorted=False
            )
        has_video = x_video is not None and len_v is not None
        if has_video:
            len_v[len_v == 0] = 1
            x_video = pack_padded_sequence(
                x_video,
                len_v.cpu().numpy(),
                batch_first=True,
                enforce_sorted=False
            )

        x_audio, _ = self.audio_rnn(x_audio)
        x_text, _ = self.text_rnn(x_text)
        if has_video:
            x_video, _ = self.video_rnn(x_video)
        if len_a[0] != 0:
            x_audio, _ = pad_packed_sequence(
                x_audio,
                batch_first=True
            )
        if len_t[0] != 0:
            x_text, _ = pad_packed_sequence(
                x_text,
                batch_first=True
            )
        if has_video and len_v[0] != 0:
            x_video, _ = pad_packed_sequence(
                x_video,
                batch_first=True
            )

        # 3. Attention
        if self.en_att:
            if self.att_name == 'multihead':
                x_audio, _ = self.audio_att(x_audio, x_audio, x_audio)
                x_text, _ = self.text_att(x_text, x_text, x_text)
                x_audio = torch.mean(x_audio, axis=1)
                x_text = torch.mean(x_text, axis=1)
                if has_video:
                    x_video, _ = self.video_att(x_video, x_video, x_video)
                    x_video = torch.mean(x_video, axis=1)
            elif self.att_name == 'base':
                x_audio = self.audio_att(x_audio)
                x_text = self.text_att(x_text, len_t)
                if has_video:
                    x_video = self.video_att(x_video, len_v)
            elif self.att_name == "fuse_base":
                a_max_len = x_audio.shape[1]
                if has_video:
                    b_max_len = x_text.shape[1]
                    x_mm = torch.cat((x_audio, x_text, x_video), dim=1)
                    x_mm = self.fuse_att(x_mm, val_a=len_a, val_b=len_t,
                                         val_c=len_v, a_len=a_max_len, b_len=b_max_len)
                elif self.num_modalities >= 3:
                    # 3-mod model but video missing → create dummy video
                    bsz = x_audio.size(0); d_hid = x_audio.size(2)
                    dummy_v = torch.zeros(bsz, 0, d_hid, device=x_audio.device)
                    dummy_len = torch.zeros(bsz, dtype=torch.long, device=x_audio.device)
                    b_max_len = x_text.shape[1]
                    x_mm = torch.cat((x_audio, x_text, dummy_v), dim=1)
                    x_mm = self.fuse_att(x_mm, val_a=len_a, val_b=len_t,
                                         val_c=dummy_len, a_len=a_max_len, b_len=b_max_len)
                else:
                    x_mm = torch.cat((x_audio, x_text), dim=1)
                    x_mm = self.fuse_att(x_mm, len_a, len_t, a_max_len)
        else:
            # 4. Average pooling Projection
            x_audio = torch.mean(x_audio, axis=1)
            x_text = torch.mean(x_text, axis=1)
            x_video_feat = torch.mean(x_video, axis=1) if has_video else (
                torch.zeros_like(x_audio) if self.num_modalities >= 3 else None)
            if x_video_feat is not None:
                x_mm = torch.cat((x_audio, x_text, x_video_feat), dim=1)
            else:
                x_mm = torch.cat((x_audio, x_text), dim=1)

        # 5. Projection
        if self.en_att and self.att_name != "fuse_base":
            if has_video:
                x_audio = self.audio_proj(x_audio)
                x_text = self.text_proj(x_text)
                x_video_feat = self.video_proj(x_video)
                x_mm = torch.cat((x_audio, x_text, x_video_feat), dim=1)
            elif self.num_modalities >= 3:
                x_audio_feat_proj = self.audio_proj(x_audio)
                x_text_feat_proj = self.text_proj(x_text)
                dummy_v = torch.zeros(x_audio.size(0), x_audio.size(1),
                                      device=x_audio.device)
                x_video_feat = self.video_proj(dummy_v)
                x_mm = torch.cat((x_audio_feat_proj, x_text_feat_proj, x_video_feat), dim=1)
                x_audio = x_audio_feat_proj
                x_text = x_text_feat_proj
            else:
                x_audio = self.audio_proj(x_audio)
                x_text = self.text_proj(x_text)
                x_mm = torch.cat((x_audio, x_text), dim=1)

        # 6. MM embedding and predict
        preds = self.classifier(x_mm)

        x_audio_feat = x_audio if x_audio.dim() == 2 else torch.mean(x_audio, axis=1)
        x_text_feat = x_text if x_text.dim() == 2 else torch.mean(x_text, axis=1)
        feat_dict = {"audio_feat": x_audio_feat,
                     "text_feat": x_text_feat}
        if has_video:
            x_video_feat = x_video if x_video.dim() == 2 else torch.mean(x_video, axis=1)
            feat_dict["video_feat"] = x_video_feat

        # 7. Cross-modal projections for FedPCD
        h = self.cross_modal_adapt(x_mm)
        for mod_name, proj_head in self.cross_modal_heads.items():
            feat_dict[f"{mod_name}_cross_proj"] = proj_head(h)

        return preds, x_mm, feat_dict


class ImageTextClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        img_input_dim: int,     # Image data input dim
        text_input_dim: int,    # Text data input dim
        d_hid: int=64,          # Hidden Layer size
        en_att: bool=False,     # Enable self attention or not
        att_name: str='',       # Attention Name
        d_head: int=6           # Head dim
    ):
        super(ImageTextClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        
        # Projection head
        self.img_proj = nn.Sequential(
            nn.Linear(img_input_dim, d_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(d_hid, d_hid)
        )
            
        # RNN module
        self.text_rnn = nn.GRU(
            input_size=text_input_dim, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        # Self attention module
        if self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )
        
        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )

        # Cross-modal projection heads for FedPCD
        if self.en_att and self.att_name == "fuse_base":
            cross_input_dim = d_hid * d_head
        else:
            cross_input_dim = d_hid * 2
        self.cross_modal_adapt = nn.Sequential(
            nn.Linear(cross_input_dim, d_hid),
            nn.ReLU()
        )
        self.cross_modal_heads = nn.ModuleDict({
            "img_feat": nn.Sequential(
                nn.Linear(d_hid, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, d_hid)
            ),
            "text_feat": nn.Sequential(
                nn.Linear(d_hid, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, d_hid)
            )
        })

        self.init_weight()
        
    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_img, x_text, len_i, len_t):
        # 1. img proj
        x_img_proj = self.img_proj(x_img[:, 0, :])

        # 2. Rnn forward (handle zero-length text = missing modality)
        text_missing = (len_t[0] == 0)
        if not text_missing:
            x_text_packed = pack_padded_sequence(
                x_text,
                len_t.cpu().numpy(),
                batch_first=True,
                enforce_sorted=False
            )
            x_text_rnn, _ = self.text_rnn(x_text_packed)
            x_text_rnn, _ = pad_packed_sequence(x_text_rnn, batch_first=True)
        else:
            # Missing text modality: create dummy RNN output
            x_text_rnn = torch.zeros(
                x_img_proj.size(0), 1, self.text_rnn.hidden_size,
                device=x_img_proj.device
            )

        # 3. Feature extraction
        if self.en_att and self.att_name == "fuse_base":
            x_mm = torch.cat((x_img_proj.unsqueeze(dim=1), x_text_rnn), dim=1)
            x_mm = self.fuse_att(x_mm, len_i, len_t, 1)
            x_img_feat = x_img_proj
            x_text_feat = torch.mean(x_text_rnn, axis=1)
        else:
            x_img_feat = x_img_proj
            x_text_feat = torch.mean(x_text_rnn, axis=1)
            x_mm = torch.cat((x_img_feat, x_text_feat), dim=1)

        # 4. Predict
        preds = self.classifier(x_mm)

        # 5. Feature dict for FedPCD distillation
        feat_dict = {
            "img_feat": x_img_feat,
            "text_feat": x_text_feat
        }

        # 6. Cross-modal projections
        h = self.cross_modal_adapt(x_mm)
        for mod_name, proj_head in self.cross_modal_heads.items():
            feat_dict[f"{mod_name}_cross_proj"] = proj_head(h)

        return preds, x_mm, feat_dict


class HARClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,       # Number of classes
        acc_input_dim: int,     # Acc data input dim
        gyro_input_dim: int,    # Gyro data input dim
        d_hid: int=128,         # Hidden Layer size
        n_filters: int=32,      # number of filters
        en_att: bool=False,     # Enable self attention or not
        att_name: str='',       # Attention Name
        d_head: int=6,          # Head dim
    ):
        super(HARClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.d_hid = d_hid
        # Conv Encoder module
        self.acc_conv = Conv1dEncoder(
            input_dim=acc_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        self.gyro_conv = Conv1dEncoder(
            input_dim=gyro_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )

        # RNN module
        self.acc_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        self.gyro_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        # Self attention module
        if self.att_name == "multihead":
            self.acc_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
            
            self.gyro_att = torch.nn.MultiheadAttention(
                embed_dim=d_hid, 
                num_heads=4, 
                dropout=self.dropout_p
            )
        elif self.att_name == "base":
            self.acc_att = BaseSelfAttention(d_hid=d_hid)
            self.gyro_att = BaseSelfAttention(d_hid=d_hid)
        elif self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )
        
        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            # 普通模式：拼接两个模态的特征
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        
        # 跨模态投影头：将全局融合表示映射到缺失模态特征空间（全局共享，参与联邦聚合）
        if self.en_att and self.att_name == "fuse_base":
            cross_input_dim = d_hid * d_head
        else:
            cross_input_dim = d_hid * 2
        self.cross_modal_adapt = nn.Sequential(
            nn.Linear(cross_input_dim, d_hid),
            nn.ReLU()
        )
        self.cross_modal_heads = nn.ModuleDict({
            "acc_feat": nn.Sequential(
                nn.Linear(d_hid, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, d_hid)
            ),
            "gyro_feat": nn.Sequential(
                nn.Linear(d_hid, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, d_hid)
            )
        })

        # Checkpoint 4C: completion adapter. Maps concatenated pooled features
        # [B, 256] -> [B, classifier_input_dim] so the SHARED main classifier
        # can consume completed features. Participates in FedAvg.
        if self.en_att and self.att_name == "fuse_base":
            self.completion_adapter = nn.Sequential(
                nn.Linear(d_hid * 2, cross_input_dim),
                nn.ReLU()
            )
        else:
            # non-fuse_base: classifier already takes [B, 256], no adapter needed
            self.completion_adapter = nn.Identity()

        # Checkpoint 4C rev2: SEPARATE auxiliary completion classifier head.
        # Avoids polluting the main classifier with completion-path gradients.
        self.completion_head = nn.Sequential(
            nn.Linear(d_hid * 2, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, num_classes)
        )

        self.init_weight()


    def init_weight(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_acc, x_gyro, l_a, l_b):
            # 1. Conv & RNN 基础特征提取
            x_acc = self.acc_conv(x_acc)
            x_gyro = self.gyro_conv(x_gyro)
            x_acc, _ = self.acc_rnn(x_acc)
            x_gyro, _ = self.gyro_rnn(x_gyro)

            # 2. 池化/注意力（得到全局表示）
            if self.en_att:
                if self.att_name == 'multihead':
                    x_acc_feat, _ = self.acc_att(x_acc, x_acc, x_acc)
                    x_gyro_feat, _ = self.gyro_att(x_gyro, x_gyro, x_gyro)
                    x_acc_feat = torch.mean(x_acc_feat, axis=1)
                    x_gyro_feat = torch.mean(x_gyro_feat, axis=1)
                elif self.att_name == 'base':
                    x_acc_feat = self.acc_att(x_acc)
                    x_gyro_feat = self.gyro_att(x_gyro)
                elif self.att_name == 'fuse_base':
                    l_a_pool = l_a // 8
                    l_b_pool = l_b // 8
                    x_mm = torch.cat((x_acc, x_gyro), dim=1)
                    x_mm = self.fuse_att(x_mm, val_a=l_a_pool, val_b=l_b_pool, a_len=x_acc.shape[1])
                    x_acc_feat = torch.mean(x_acc, axis=1)
                    x_gyro_feat = torch.mean(x_gyro, axis=1)
                else:
                    x_acc_feat = torch.mean(x_acc, axis=1)
                    x_gyro_feat = torch.mean(x_gyro, axis=1)
            else:
                x_acc_feat = torch.mean(x_acc, axis=1)
                x_gyro_feat = torch.mean(x_gyro, axis=1)

            # 3. 融合
            if self.en_att and self.att_name == "fuse_base":
                pass  # x_mm已在上面计算
            else:
                x_mm = torch.cat((x_acc_feat, x_gyro_feat), dim=1)

            # 4. 预测
            preds = self.classifier(x_mm)

            # 特征字典（用于双蒸馏等扩展）
            feat_dict = {
                "acc_feat": x_acc_feat,
                "gyro_feat": x_gyro_feat
            }

            # 跨模态投影结果（全局表示 → 各模态空间），放入 feat_dict 中方便传递
            h = self.cross_modal_adapt(x_mm)
            for mod_name, proj_head in self.cross_modal_heads.items():
                feat_dict[f"{mod_name}_cross_proj"] = proj_head(h)

            return preds, x_mm, feat_dict

    # ==================== FedPCD-New: leak-free target-modality prediction ====================
    # These methods are ONLY used by the new leak-free path (flag --leak_free_cross).
    # forward() above is left untouched so FedPCD-Old stays byte-identical.

    def _encode_seq(self, x_acc, x_gyro):
        """Conv+RNN encoders -> per-modality sequence features [B, T, d_hid]."""
        xa = self.acc_conv(x_acc)
        xg = self.gyro_conv(x_gyro)
        xa, _ = self.acc_rnn(xa)
        xg, _ = self.gyro_rnn(xg)
        return xa, xg

    def _fuse_masked(self, xa, xg, va, vb):
        """fuse_base fusion with per-modality valid lengths va, vb (already pooled // 8).
        A modality with valid length 0 is fully masked out of the attention."""
        x_mm = torch.cat((xa, xg), dim=1)
        x_mm = self.fuse_att(x_mm, val_a=va, val_b=vb, a_len=xa.shape[1])
        return x_mm

    def predict_target_features(self, x_acc, x_gyro, l_a, l_b):
        """Leak-free target-modality feature prediction.

        To predict modality m, the target modality is REMOVED from the fusion input
        (encoded tokens zeroed AND attention length set to 0); only the other
        (source) modality drives the prediction. This guarantees that the real
        target modality cannot leak into its own prediction.

          acc_target_pred  = P_acc ( adapt( fuse( 0 , gyro ) ) )   # source = gyro
          gyro_target_pred = P_gyro( adapt( fuse( acc , 0 ) ) )    # source = acc

        Returns dict {'acc_feat': acc_target_pred, 'gyro_feat': gyro_target_pred},
        each [B, d_hid]. The caller decides which prediction is usable based on
        which source modality is actually present.
        """
        xa, xg = self._encode_seq(x_acc, x_gyro)
        zeros_a = torch.zeros_like(xa)
        zeros_b = torch.zeros_like(xg)
        va = (l_a // 8)
        vb = (l_b // 8)
        zlen = torch.zeros_like(va)

        # predict ACC target: mask acc (zeros + len 0), source = gyro
        x_mm_no_a = self._fuse_masked(zeros_a, xg, zlen, vb)
        acc_target = self.cross_modal_heads['acc_feat'](self.cross_modal_adapt(x_mm_no_a))

        # predict GYRO target: mask gyro (zeros + len 0), source = acc
        x_mm_no_b = self._fuse_masked(xa, zeros_b, va, zlen)
        gyro_target = self.cross_modal_heads['gyro_feat'](self.cross_modal_adapt(x_mm_no_b))

        return {'acc_feat': acc_target, 'gyro_feat': gyro_target}

    # ==================== Checkpoint 4C: task-coupled completion ====================
    def classify_completed_features(self, acc_feat, gyro_feat):
        """Classify from (potentially predicted) pooled features [B, 128] each.
        Concatenates [acc, gyro] -> INDEPENDENT completion_head (not the main
        classifier). This isolates task_rec gradients from the main decision path.
        Feature order is FIXED: always [acc, gyro], regardless of which is real.
        """
        completed = torch.cat([acc_feat, gyro_feat], dim=-1)   # [B, 256]
        return self.completion_head(completed)

    def extract_completion_features(self, x_acc, x_gyro, l_a, l_b):
        """One-pass: forward() features + leak-free target predictions.
        Avoids double-encoding (_encode_seq called once here).
        Returns dict with real pooled features and predicted targets.
        """
        xa, xg = self._encode_seq(x_acc, x_gyro)
        # pooled real features (from actual encoder outputs, BEFORE masking)
        acc_real = torch.mean(xa, dim=1)   # [B, 128]
        gyr_real = torch.mean(xg, dim=1)   # [B, 128]

        zeros_a = torch.zeros_like(xa); zeros_b = torch.zeros_like(xg)
        va = (l_a // 8); vb = (l_b // 8); zlen = torch.zeros_like(va)

        acc_pred = self.cross_modal_heads['acc_feat'](
            self.cross_modal_adapt(self._fuse_masked(zeros_a, xg, zlen, vb)))
        gyr_pred = self.cross_modal_heads['gyro_feat'](
            self.cross_modal_adapt(self._fuse_masked(xa, zeros_b, va, zlen)))

        return {
            'acc_real': acc_real, 'gyro_real': gyr_real,
            'acc_pred': acc_pred, 'gyro_pred': gyr_pred,
        }


class ECGClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,           # Number of classes 
        i_to_avf_input_dim: int,    # 6 lead ecg
        v1_to_v6_input_dim: int,    # v1-v6 ecg
        d_hid: int=64,              # Hidden Layer size
        n_filters: int=32,          # number of filters
        en_att: bool=False,         # Enable self attention or not
        att_name: str='',       # Attention Name
        d_head: int=6           # Head dim
    ):
        super(ECGClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.d_hid = d_hid
        
        # Conv Encoder module
        self.i_to_avf_conv = Conv1dEncoder(
            input_dim=i_to_avf_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        self.v1_to_v6_conv = Conv1dEncoder(
            input_dim=v1_to_v6_input_dim, 
            n_filters=n_filters, 
            dropout=self.dropout_p, 
        )
        
        # RNN module
        self.i_to_avf_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        self.v1_to_v6_rnn = nn.GRU(
            input_size=n_filters*4, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        # Self attention module
        if self.att_name == "base":
            self.i_to_avf_att = BaseSelfAttention(d_hid=d_hid)
            self.v1_to_v6_att = BaseSelfAttention(d_hid=d_hid)
        elif self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )

        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            # Classifier head
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        
        # 跨模态投影头：将全局融合表示映射到缺失模态特征空间（全局共享，参与联邦聚合）
        if self.en_att and self.att_name == "fuse_base":
            cross_input_dim = d_hid * d_head
        else:
            cross_input_dim = d_hid * 2
        self.cross_modal_adapt = nn.Sequential(
            nn.Linear(cross_input_dim, d_hid),
            nn.ReLU()
        )
        self.cross_modal_heads = nn.ModuleDict({
            "i_to_avf_feat": nn.Sequential(
                nn.Linear(d_hid, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, d_hid)
            ),
            "v1_to_v6_feat": nn.Sequential(
                nn.Linear(d_hid, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, d_hid)
            )
        })

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_i_to_avf, x_v1_to_v6, l_a, l_b):
        # 1. Conv & RNN 基础特征提取
        x_i_to_avf = self.i_to_avf_conv(x_i_to_avf)
        x_v1_to_v6 = self.v1_to_v6_conv(x_v1_to_v6)
        x_i_to_avf, _ = self.i_to_avf_rnn(x_i_to_avf)
        x_v1_to_v6, _ = self.v1_to_v6_rnn(x_v1_to_v6)

        # 2. 池化/注意力（得到全局表示）
        if self.en_att:
            if self.att_name == 'base':
                x_i_to_avf_feat = self.i_to_avf_att(x_i_to_avf)
                x_v1_to_v6_feat = self.v1_to_v6_att(x_v1_to_v6)
            elif self.att_name == 'fuse_base':
                l_a_pool = l_a // 8
                l_b_pool = l_b // 8
                x_mm = torch.cat((x_i_to_avf, x_v1_to_v6), dim=1)
                x_mm = self.fuse_att(x_mm, val_a=l_a_pool, val_b=l_b_pool, a_len=x_i_to_avf.shape[1])
                x_i_to_avf_feat = torch.mean(x_i_to_avf, axis=1)
                x_v1_to_v6_feat = torch.mean(x_v1_to_v6, axis=1)
            else:
                x_i_to_avf_feat = torch.mean(x_i_to_avf, axis=1)
                x_v1_to_v6_feat = torch.mean(x_v1_to_v6, axis=1)
        else:
            x_i_to_avf_feat = torch.mean(x_i_to_avf, axis=1)
            x_v1_to_v6_feat = torch.mean(x_v1_to_v6, axis=1)

        # 3. 融合
        if self.en_att and self.att_name == "fuse_base":
            pass  # x_mm已在上面计算
        else:
            x_mm = torch.cat((x_i_to_avf_feat, x_v1_to_v6_feat), dim=1)

        # 4. 预测
        preds = self.classifier(x_mm)

        feat_dict = {"i_to_avf_feat": x_i_to_avf_feat,
                     "v1_to_v6_feat": x_v1_to_v6_feat}
        
        # 跨模态投影结果（全局表示 → 各模态空间），放入 feat_dict 中方便传递
        h = self.cross_modal_adapt(x_mm)
        for mod_name, proj_head in self.cross_modal_heads.items():
            feat_dict[f"{mod_name}_cross_proj"] = proj_head(h)
        
        return preds, x_mm, feat_dict


class Conv1dEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int, 
        n_filters: int,
        dropout: float=0.1
    ):
        super().__init__()
        # conv module
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
            self,
            x: Tensor   # shape => [batch_size (B), num_data (T), feature_dim (D)]
        ):
        x = x.float()
        x = x.permute(0, 2, 1)
        # conv1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x
    
    
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    

class AdditiveAttention(nn.Module):
    def __init__(
        self, 
        d_hid:  int=64, 
        d_att:  int=256
    ):
        super().__init__()

        self.query_proj = nn.Linear(d_hid, d_att, bias=False)
        self.key_proj = nn.Linear(d_hid, d_att, bias=False)
        self.bias = nn.Parameter(torch.rand(d_att).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(d_att, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, 
        query: Tensor,
        key: Tensor, 
        value: Tensor,
        valid_lens: Tensor
    ):
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = masked_softmax(scores, valid_lens)
        attn = self.dropout(attn)
        output = torch.bmm(attn.unsqueeze(1), value)
        return output
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            pdb.set_trace()
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class HirarchicalAttention(nn.Module):
    def __init__(self, d_hid: int):
        super(HirarchicalAttention, self).__init__()
        self.w_linear = nn.Linear(d_hid, d_hid)
        self.u_w = nn.Linear(d_hid, 1, bias=False)

    def forward(self, input: torch.Tensor):
        u_it = torch.tanh(self.w_linear(input))
        a_it = torch.softmax(self.u_w(u_it), dim=1)
        s_i = input * a_it
        return s_i


class BaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
        self, 
        d_hid:  int=64
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 1)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(1, 1)

    def forward(
        self,
        x: Tensor,
        val_l=None
    ):
        att = self.att_pool(self.att_fc1(x))
        att = self.att_fc2(att).squeeze(-1)
        if val_l is not None:
            for idx in range(len(val_l)):
                att[idx, val_l[idx]:] = -1e6
        att = torch.softmax(att, dim=1)
        x = (att.unsqueeze(2) * x).sum(axis=1)
        return x
    
class FuseBaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
        self, 
        d_hid:  int=64,
        d_head: int=4
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(512, d_head)

        self.d_hid = d_hid
        self.d_head = d_head

    def forward(
        self,
        x: Tensor,
        val_a=None,
        val_b=None,
        a_len=None
    ):
        att = self.att_pool(self.att_fc1(x))
        att = self.att_fc2(att)
        att = att.transpose(1, 2)
        if val_a is not None:
            for idx in range(len(val_a)):
                att[idx, :, val_a[idx]:a_len] = -1e5
                att[idx, :, a_len+val_b[idx]:] = -1e5
        att = torch.softmax(att, dim=2)
        x = torch.matmul(att, x)
        x = x.reshape(x.shape[0], self.d_head*self.d_hid)
        return x


class FuseBaseSelfAttention3Mod(nn.Module):
    """3-modality version of FuseBaseSelfAttention for acc+gyro+mag fusion."""
    def __init__(self, d_hid: int = 64, d_head: int = 4):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(512, d_head)
        self.d_hid = d_hid
        self.d_head = d_head

    def forward(self, x, val_a=None, val_b=None, val_c=None,
                a_len=None, b_len=None):
        att = self.att_pool(self.att_fc1(x))
        att = self.att_fc2(att)
        att = att.transpose(1, 2)
        if val_a is not None:
            offset_b = a_len
            offset_c = a_len + b_len
            for idx in range(len(val_a)):
                att[idx, :, val_a[idx]:offset_b] = -1e5
                att[idx, :, offset_b + val_b[idx]:offset_c] = -1e5
                att[idx, :, offset_c + val_c[idx]:] = -1e5
        att = torch.softmax(att, dim=2)
        x = torch.matmul(att, x)
        x = x.reshape(x.shape[0], self.d_head * self.d_hid)
        return x


class MHealthClassifier(nn.Module):
    """3-modality HAR classifier for mHealth: acc + gyro + mag from right wrist."""

    def __init__(
        self,
        num_classes: int,
        acc_input_dim: int,
        gyro_input_dim: int,
        mag_input_dim: int,
        d_hid: int = 128,
        n_filters: int = 32,
        en_att: bool = False,
        att_name: str = '',
        d_head: int = 6
    ):
        super(MHealthClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.d_hid = d_hid

        self.acc_conv = Conv1dEncoder(acc_input_dim, n_filters, dropout=self.dropout_p)
        self.gyro_conv = Conv1dEncoder(gyro_input_dim, n_filters, dropout=self.dropout_p)
        self.mag_conv = Conv1dEncoder(mag_input_dim, n_filters, dropout=self.dropout_p)

        self.acc_rnn = nn.GRU(n_filters * 4, d_hid, num_layers=1, batch_first=True,
                              dropout=self.dropout_p, bidirectional=False)
        self.gyro_rnn = nn.GRU(n_filters * 4, d_hid, num_layers=1, batch_first=True,
                               dropout=self.dropout_p, bidirectional=False)
        self.mag_rnn = nn.GRU(n_filters * 4, d_hid, num_layers=1, batch_first=True,
                              dropout=self.dropout_p, bidirectional=False)

        if self.att_name == "multihead":
            self.acc_att = torch.nn.MultiheadAttention(d_hid, num_heads=4, dropout=self.dropout_p)
            self.gyro_att = torch.nn.MultiheadAttention(d_hid, num_heads=4, dropout=self.dropout_p)
            self.mag_att = torch.nn.MultiheadAttention(d_hid, num_heads=4, dropout=self.dropout_p)
        elif self.att_name == "base":
            self.acc_att = BaseSelfAttention(d_hid=d_hid)
            self.gyro_att = BaseSelfAttention(d_hid=d_hid)
            self.mag_att = BaseSelfAttention(d_hid=d_hid)
        elif self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention3Mod(d_hid=d_hid, d_head=d_head)

        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_hid * 3, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )

        if self.en_att and self.att_name == "fuse_base":
            cross_input_dim = d_hid * d_head
        else:
            cross_input_dim = d_hid * 3
        self.cross_modal_adapt = nn.Sequential(
            nn.Linear(cross_input_dim, d_hid),
            nn.ReLU()
        )
        self.cross_modal_heads = nn.ModuleDict({
            "acc_feat": nn.Sequential(
                nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Linear(d_hid, d_hid)),
            "gyro_feat": nn.Sequential(
                nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Linear(d_hid, d_hid)),
            "mag_feat": nn.Sequential(
                nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Linear(d_hid, d_hid))
        })

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_acc, x_gyro, x_mag, l_a, l_b, l_c):
        x_acc = self.acc_conv(x_acc)
        x_gyro = self.gyro_conv(x_gyro)
        x_mag = self.mag_conv(x_mag)
        x_acc, _ = self.acc_rnn(x_acc)
        x_gyro, _ = self.gyro_rnn(x_gyro)
        x_mag, _ = self.mag_rnn(x_mag)

        if self.en_att:
            if self.att_name == 'multihead':
                x_acc_feat, _ = self.acc_att(x_acc, x_acc, x_acc)
                x_gyro_feat, _ = self.gyro_att(x_gyro, x_gyro, x_gyro)
                x_mag_feat, _ = self.mag_att(x_mag, x_mag, x_mag)
                x_acc_feat = torch.mean(x_acc_feat, axis=1)
                x_gyro_feat = torch.mean(x_gyro_feat, axis=1)
                x_mag_feat = torch.mean(x_mag_feat, axis=1)
            elif self.att_name == 'base':
                x_acc_feat = self.acc_att(x_acc)
                x_gyro_feat = self.gyro_att(x_gyro)
                x_mag_feat = self.mag_att(x_mag)
            elif self.att_name == 'fuse_base':
                l_a_pool = l_a // 8
                l_b_pool = l_b // 8
                l_c_pool = l_c // 8
                x_mm = torch.cat((x_acc, x_gyro, x_mag), dim=1)
                x_mm = self.fuse_att(x_mm, val_a=l_a_pool, val_b=l_b_pool,
                                     val_c=l_c_pool,
                                     a_len=x_acc.shape[1], b_len=x_gyro.shape[1])
                x_acc_feat = torch.mean(x_acc, axis=1)
                x_gyro_feat = torch.mean(x_gyro, axis=1)
                x_mag_feat = torch.mean(x_mag, axis=1)
            else:
                x_acc_feat = torch.mean(x_acc, axis=1)
                x_gyro_feat = torch.mean(x_gyro, axis=1)
                x_mag_feat = torch.mean(x_mag, axis=1)
        else:
            x_acc_feat = torch.mean(x_acc, axis=1)
            x_gyro_feat = torch.mean(x_gyro, axis=1)
            x_mag_feat = torch.mean(x_mag, axis=1)

        if self.en_att and self.att_name == "fuse_base":
            pass
        else:
            x_mm = torch.cat((x_acc_feat, x_gyro_feat, x_mag_feat), dim=1)

        preds = self.classifier(x_mm)

        feat_dict = {
            "acc_feat": x_acc_feat,
            "gyro_feat": x_gyro_feat,
            "mag_feat": x_mag_feat
        }

        h = self.cross_modal_adapt(x_mm)
        for mod_name, proj_head in self.cross_modal_heads.items():
            feat_dict[f"{mod_name}_cross_proj"] = proj_head(h)

        return preds, x_mm, feat_dict