import copy
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .datasets import register_dataset
from .data_utils import truncate_feats

'''
SAMM 数据集特征由VideoMAE抽取，步长为4，一次提取16帧，tensor [seq_len, 768]
seg / 4
[1 ~ 8]  [9 ~ 16]
   1        2        3
[1 ~ 16] [5 ~ 20] [9 ~ 24] [13, 28] [17, 32]
'''


@register_dataset("samm")
class SAMMDataset(Dataset):
    def __init__(
            self,
            is_training,  # if in training mode
            split,  # split, a tuple/list allowing concat of subsets
            feat_folder,  # folder for features
            excel_file,  # excel file for annotations
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
        self.excel_file = excel_file

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
        dict_db = self._load_excel_db(self.excel_file)
        self.data_list = dict_db

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'samm',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_excel_db(self, excel_file):
        df_duration = pd.read_csv("labels/samm_duration.csv")
        dic_duration = {}
        for idx in range(len(df_duration)):
            dic_duration[df_duration['sub_name'][idx]] = int(df_duration['duration'][idx])


        df = pd.read_excel(excel_file, engine='openpyxl')  # 对于.xlsx文件，使用'openpyxl'引擎
        dict_list = []
        for idx in range(len(df)):
            exp_type = df['Type'][idx]
            if exp_type == "Macro":
                exp_type = 0
            elif exp_type == "Micro - 1/2":
                exp_type = 1
            Offset = int(df['Offset'][idx])
            Onset = int(df['Onset'][idx])
            duration = int(df['Duration'][idx])
            fileName = df['Filename'][idx].split("_")
            video_name = fileName[0] + "_" + fileName[1]
            if len(dict_list) == 0 or dict_list[-1]["name"] != video_name:
                temp_dic = {
                    "name": video_name,
                    "segments": [[Onset, Offset]],
                    "labels": [exp_type],
                    "duration": dic_duration["samm_" + video_name]
                }
                dict_list.append(temp_dic)
            elif dict_list[-1]["name"] == video_name:
                dict_list[-1]["segments"].append([Onset, Offset])
                dict_list[-1]["labels"].append(exp_type)
        if self.split[0] == 'test':
            dict_list = []
            test_df = pd.read_csv("labels/samm_test.csv")
            for idx in range(len(test_df)):
                dict_list.append({
                    "name": test_df["sub_name"][idx],
                    "duration": test_df["duration"][idx]
                })
        return dict_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]
        # load features
        filename = os.path.join(self.feat_folder,
                                "samm_" + video_item['name'] + self.file_ext)
        if self.split[0] == "test":
            filename = os.path.join("/data2/zyzhang/dataset/SAMM_feature_test", video_item['name'] + self.file_ext)
        
        feats = np.load(filename).astype(np.float32)
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 1.875

        if self.split[0] == "test":
            data_dict = {'video_id': video_item['name'],
                         'feats': feats,  # C x T
                         'duration': video_item['duration']}
            return data_dict

        segments = torch.from_numpy((np.array(video_item['segments'], dtype=np.float32) - 8.5) / feat_stride)
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
