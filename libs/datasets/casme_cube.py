import copy
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .datasets import register_dataset
from .data_utils import truncate_feats


@register_dataset("casme")
class CASMEDataset(Dataset):
    def __init__(
            self,
            is_training,  # if in training mode
            split,  # split, a tuple/list allowing concat of subsets
            feat_folder,  # folder for features
            csv_file,  # excel file for annotations
            feat_stride,  # temporal stride of the feats
            num_frames,  # number of frames for each feat
            default_fps,  # default fps
            downsample_rate,  # downsample rate for feats
            max_seq_len,  # maximum sequence length during training
            trunc_thresh,  # threshold for truncate an action segment
            crop_ratio,  # a tuple (e.g., (0.9, 1.0)) for random cropping
            input_dim,  # input feat dim
            num_classes,  # number of action categories
            file_prefix,  # feature file prefix if any
            file_ext,  # feature file extension if any
            force_upsampling  # force to upsample to max_seq_len
    ):
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.csv_file = csv_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db = self._load_csv_db(self.csv_file)
        self.data_list = dict_db

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'casme',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_csv_db(self, csv_file):

        duration_path = "/data2/zyzhang/MEGC/MEGC_2024_code/labels/start_end.npy"
        file_path = "/data2/zyzhang/MEGC/MEGC_2024_code/labels/CAS(ME)3_part_A_v2.xls"
        duration = np.load(duration_path, allow_pickle=True).item()
        df = pd.read_excel(file_path, sheet_name=0)
        dic_seg = {}
        for idx in range(len(df)):
            name = df["Subject"][idx] + "_" + df["Filename"][idx].lower()
            dic_seg[name] = {
                "name": name,
                "segments": [],
                "start": duration[name][0] - 1,
                "duration": duration[name][1],
                "labels": [],
            }
        for idx in range(len(df)):
            name = df["Subject"][idx] + "_" + df["Filename"][idx].lower()
            dic_seg[name]["segments"].append([df["Onset"][idx] - dic_seg[name]["start"] + 1, df["Offset"][idx] - dic_seg[name]["start"] + 1])
            if df["Expression type"][idx] == "Macro-expression":
                dic_seg[name]["labels"].append(0)
            else:
                dic_seg[name]["labels"].append(1)

        dict_list = []
        extend_train_list = ["spNO.3", "spNO.4", "spNO.5", "spNO.6"]
        extend_valid_list = ["spNO.1", "spNO.2", "spNO.176", "spNO.177"]
        k_list = list(dic_seg.keys())
        for feature_name in os.listdir("/data2/zyzhang/dataset/CAS_feature_train"):
            if feature_name.split("_")[0] in extend_train_list and feature_name.replace(".npy", "") in k_list:
                dic_seg[feature_name.replace(".npy", "")]["feature_path"] = "/data2/zyzhang/dataset/CAS_feature_train/" + feature_name
                # dict_list.append(dic_seg[feature_name.replace(".npy", "")])
            
        df_duration = pd.read_csv("labels/cas_duration.csv")
        dic_duration = {}
        dic_org_name = {}
        for idx in range(len(df_duration)):
            new_name = df_duration['vid'][idx][:6] + "0" + df_duration['vid'][idx][6:]
            dic_duration[new_name] = int(df_duration['duration'][idx])
            dic_org_name[new_name] = df_duration['org_vid'][idx]
        df = pd.read_csv(csv_file)
        
        for idx in range(len(df)):
            exp_type = df['type_idx'][idx] - 1
            Offset = int(df['end_frame'][idx])
            Onset = int(df['start_frame'][idx])
            video_name = df['video_name'][idx]
            if len(dict_list) == 0 or dict_list[-1]["name"] != video_name:
                temp_dic = {
                    "name": video_name,
                    "segments": [[Onset, Offset]],
                    "labels": [exp_type],
                    "duration": dic_duration[video_name],
                    "feature_path": os.path.join(self.feat_folder, dic_org_name[video_name] + self.file_ext)
                }
                dict_list.append(temp_dic)
            elif dict_list[-1]["name"] == video_name:
                dict_list[-1]["segments"].append([Onset, Offset])
                dict_list[-1]["labels"].append(exp_type)
        train_dict_list = dict_list
        if self.split[0] == "valid":
            valid_dict_list = []
            for feature_name in os.listdir("/data2/zyzhang/dataset/CAS_feature_train"):
                if feature_name.split("_")[0] in extend_valid_list and feature_name.replace(".npy", "") in k_list:
                    dic_seg[feature_name.replace(".npy", "")]["feature_path"] = "/data2/zyzhang/dataset/CAS_feature_train/" + feature_name
                    valid_dict_list.append(dic_seg[feature_name.replace(".npy", "")])           
            return valid_dict_list
        elif self.split[0] == 'test':
            dict_list = []
            test_df = pd.read_csv("labels/cas_test_duration.csv")
            for idx in range(len(test_df)):
                dict_list.append({
                    "name": test_df["vid"][idx],
                    "duration": test_df["duration"][idx]
                })
            return dict_list
        return train_dict_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]
        # load features
        
        if self.split[0] == "test":
            filename = os.path.join("/data2/zyzhang/dataset/CAS_feature_test", video_item['name'] + self.file_ext)
        else:
            filename = video_item['feature_path']
        feats = np.load(filename).astype(np.float32)
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        if self.split[0] == "test":
            data_dict = {'video_id': video_item['name'],
                         'feats': feats,  # C x T
                         'duration': video_item['duration']}
            return data_dict

        segments = torch.from_numpy(np.array(video_item['segments'], dtype=np.float32) / feat_stride - feat_offset)
        segments = torch.clamp(segments, min=0)  
        labels = torch.from_numpy(np.array(video_item['labels'], dtype=np.int64))
        
        # return a data dict
        data_dict = {'video_id': video_item['name'],
                     'feats': feats,  # C x T
                     'segments': segments,  # N x 2
                     'labels': labels,
                     'duration': video_item['duration']}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict