# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 15:08
# @Author  : jiamingNo1
# @Email   : jiaming19.huang@foxmail.com
# @File    : feeder.py
# @Software: PyCharm
import cv2
import pickle
import numpy as np
import torch
import torch.utils.data

try:
    from data import tools
except:
    import tools


class Feeder(torch.utils.data.Dataset):
    ''' Feeder for skeleton-based action recognition
    Argument:
        data_path: the path to '.npy' data
        label_path: the path to '.pkl' label
        random_rotate: If more than 0, randomly rotate theta angel
        debug: If true, only use the first 100 samples
        mmap: If true, store data in memory
    '''

    def __init__(self,
                 data_path,
                 label_path,
                 random_rotate=0,
                 debug=False,
                 mmap=True):
        self.data_path = data_path
        self.label_path = label_path
        self.random_rotate = random_rotate
        self.debug = debug
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
        if self.debug:
            self.sample_name = self.sample_name[0:100]
            self.label = self.label[0:100]
            self.data = self.data[0:100]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data = np.array(self.data[index])
        label = np.array(self.label[index])
        if self.random_rotate > 0:
            data = tools.random_rotate(data, self.random_rotate)

        data = np.transpose(data, [1, 3, 2, 0])  # T,M,V,C
        data_reshape = np.reshape(data, (data.shape[0], 150))
        zero_row = []
        for idx in range(len(data_reshape)):
            if (data_reshape[idx, :] == np.zeros((1, 150))).all():
                zero_row.append(idx)
        data_reshape = np.delete(data_reshape, zero_row, axis=0)
        if (data_reshape[:, 0:75] == np.zeros((data_reshape.shape[0], 75))).all():
            data_reshape = np.delete(data_reshape, range(75), axis=1)
        elif (data_reshape[:, 75:150] == np.zeros((data_reshape.shape[0], 75))).all():
            data_reshape = np.delete(data_reshape, range(75, 150), axis=1)

        # min_val, max_val = [-3.602826, -2.716611, 0.]), [3.635367, 1.888282, 5.209939]
        min_val, max_val = -3.602826, 5.209939
        data_reshape = np.floor(255 * (data_reshape - min_val) / (max_val - min_val))

        rgb_data = np.reshape(data_reshape, (data_reshape.shape[0], data_reshape.shape[1] // 3, 3))
        rgb_data = cv2.resize(rgb_data, (224, 224))
        rgb_data[:, :, 0] -= 110
        rgb_data[:, :, 1] -= 110
        rgb_data[:, :, 2] -= 110
        rgb_data = np.transpose(rgb_data, [2, 0, 1])

        return rgb_data, label

    def seq_transformation(self, data):
        pass


def fetch_dataloader(mode, params):
    if 'CV' in params['dataset_name']:
        params['train_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cv/train_data.npy'
        params['train_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cv/train_label.pkl'
        params['val_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cv/val_data.npy'
        params['val_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cv/val_label.pkl'
        params['test_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cv/test_data.npy'
        params['test_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cv/test_label.pkl'
    if 'CS' in params['dataset_name']:
        params['train_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cs/train_data.npy'
        params['train_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cs/train_label.pkl'
        params['val_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cs/val_data.npy'
        params['val_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cs/val_label.pkl'
        params['test_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cs/test_data.npy'
        params['test_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB+D' + '/cs/test_label.pkl'
    if mode == 'train':
        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params['train_feeder_args']),
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=params['num_workers'],
            pin_memory=False)
    if mode == 'val':
        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params['val_feeder_args']),
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=params['num_workers'],
            pin_memory=False)
    if mode == 'test':
        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params['test_feeder_args']),
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=params['num_workers'],
            pin_memory=False)

    return loader


if __name__ == '__main__':
    data_path = 'data/NTU-RGB+D/cv/val_data.npy'
    label_path = 'data/NTU-RGB+D/cv/val_label.pkl'
    dataset = Feeder(data_path,
                     label_path,
                     random_rotate=0,
                     debug=False,
                     mmap=True)
    print(np.bincount(dataset.label))
