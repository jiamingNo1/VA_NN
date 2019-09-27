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

        max_vals, min_vals = list(), list()
        for ske_data in self.data:
            max_val = ske_data.max()
            min_val = ske_data.min()
            max_vals.append(float(max_val))
            min_vals.append(float(min_val))
        max_vals, min_vals = np.array(max_vals), np.array(min_vals)
        print('max_val: %f, min_val: %f' % (max_vals.max(), min_vals.min()))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data = np.array(self.data[index])
        label = np.array(self.label[index])

        zero_row = []
        for idx in range(len(data)):
            if (data[idx, :] == np.zeros((1, 150))).all():
                zero_row.append(idx)
        data = np.delete(data, zero_row, axis=0)
        if (data[:, 0:75] == np.zeros((data.shape[0], 75))).all():
            data = np.delete(data, range(75), axis=1)
        elif (data[:, 75:150] == np.zeros((data.shape[0], 75))).all():
            data = np.delete(data, range(75, 150), axis=1)

        # for ntu
        min_val, max_val = -4.765629, 5.187813
        data = np.floor(255 * (data - min_val) / (max_val - min_val))

        rgb_data = np.reshape(data, (data.shape[0], data.shape[1] // 3, 3))
        rgb_data = cv2.resize(rgb_data, (224, 224))
        rgb_data[:, :, 0] -= 110
        rgb_data[:, :, 1] -= 110
        rgb_data[:, :, 2] -= 110
        rgb_data = np.transpose(rgb_data, [2, 0, 1])

        return rgb_data, label

def random_rotate(data, rand_rotate):
    C, T, V, M = data.shape
    R = np.eye(3)
    for i in range(3):
        theta = (np.random.rand() * 2 - 1) * rand_rotate * np.pi
        Ri = np.zeros(3, 3)
        Ri[i, i] = 1
        Ri[(i + 1) % 3, (i + 1) % 3] = np.cos(theta)
        Ri[(i + 2) % 3, (i + 2) % 3] = np.cos(theta)
        Ri[(i + 1) % 3, (i + 2) % 3] = np.sin(theta)
        Ri[(i + 2) % 3, (i + 1) % 3] = -np.sin(theta)
        R = np.matmul(R, Ri)
    data = np.matmul(R, data.reshape(C, T * V * M)).reshape((C, T, V, M)).astype('float32')

    return data


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
                     mmap=True)
    print(np.bincount(dataset.label))
