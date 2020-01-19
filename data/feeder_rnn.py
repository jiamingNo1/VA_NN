# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 下午4:45
# @Author  : jiamingNo1
# @Email   : jiaming19.huang@foxmail.com
# @File    : feeder_rnn.py
# @Software: PyCharm
import pickle
import numpy as np
import torch
import torch.utils.data


class Feeder(torch.utils.data.Dataset):
    ''' Feeder for skeleton-based action recognition
    Argument:
        data_path: the path to '.npy' data
        label_path: the path to '.pkl' label
        mmap: If true, store data in memory
    '''

    def __init__(self,
                 data_path,
                 label_path,
                 mmap=True):
        self.data_path = data_path
        self.label_path = label_path
        self.mmap = mmap
        self.load_data()

    def load_data(self):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        # load data
        if self.mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data = np.array(self.data[index])
        label = np.array(self.label[index])
        return data, label


def fetch_dataloader(mode, params):
    if 'cv' in params['dataset_name']:
        params['train_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cv/train_data.npy'
        params['train_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cv/train_label.pkl'
        params['val_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cv/val_data.npy'
        params['val_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cv/val_label.pkl'
        params['test_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cv/test_data.npy'
        params['test_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cv/test_label.pkl'
    if 'cs' in params['dataset_name']:
        params['train_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cs/train_data.npy'
        params['train_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cs/train_label.pkl'
        params['val_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cs/val_data.npy'
        params['val_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cs/val_label.pkl'
        params['test_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cs/test_data.npy'
        params['test_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cs/test_label.pkl'

    if mode == 'train':
        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params['train_feeder_args']),
            batch_size=256,
            shuffle=True,
            num_workers=params['num_workers'],
            pin_memory=False)
    if mode == 'val':
        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params['val_feeder_args']),
            batch_size=256,
            shuffle=False,
            num_workers=params['num_workers'],
            pin_memory=False)
    if mode == 'test':
        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params['test_feeder_args']),
            batch_size=256,
            shuffle=False,
            num_workers=params['num_workers'],
            pin_memory=False)

    return loader
