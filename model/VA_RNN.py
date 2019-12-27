# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 10:40
# @Author  : jiamingNo1
# @Email   : jiaming19.huang@foxmail.com
# @File    : VA_RNN.py
# @Software: PyCharm
import math
import torch
import torch.nn as nn


class VARNN(nn.Module):
    '''
    Input shape should be (N,C,T,V)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints.
    '''

    def __int__(self, num_class=60, n_hid=100):
        super(VARNN, self).__init__()
        self.num_class = num_class
        self.rotation_lstm = nn.LSTM(3 * 25, n_hid, batch_first=True, dropout=0.5)
        self.rotation_fc = nn.Linear(n_hid, 3)
        self.translation_lstm = nn.LSTM(3 * 25, n_hid, batch_first=True, dropout=0.5)
        self.translation_fc = nn.Linear(n_hid, 3)
        self.main_lstm = nn.LSTM(3 * 25, n_hid, 3, batch_first=True, dropout=0.5)
        self.main_fc = nn.Linear(n_hid, num_class)

    def forward(self, x, target=None):
        N, C, T, V = x.size()

        out = x.permute(0, 2, 1, 3)
        out = self.rotation_lstm(out.contiguous().view(N, T, -1))
        rotation = self.rotation_lstm(out)
        rotation = self.rotation_fc(rotation[:, :, -1].squeeze())
        translation = self.translation_lstm(out)
        translation = self.translation_fc(translation[:, :, -1].squeeze())
        sub_out = []
        for n in range(N):
            rotation_x = torch.tensor([[1, 0, 0],
                                       [0, math.cos(rotation[n, 0].item()), math.sin(rotation[n, 0].item())],
                                       [0, math.sin(-rotation[n, 0].item()), math.cos(rotation[n, 0].item())]])
            rotation_y = torch.tensor(
                [[math.cos(rotation[n, 1].item()), 0, math.sin(-rotation[n, 1].item())],
                 [0, 1, 0],
                 [math.sin(rotation[n, 1].item()), 0, math.cos(rotation[n, 1].item())]])
            rotation_z = torch.tensor(
                [[math.cos(rotation[n, 2].item()), math.sin(rotation[n, 2].item()), 0],
                 [math.sin(-rotation[n, 2].item()), math.cos(rotation[n, 2].item()), 0],
                 [0, 0, 1]])
            trans = x[n, :, :, :] - torch.tensor(translation[n, :]).view(1, -1, 1, 1).expand(1, C, T, V)
            trans = trans.view(C, T * V)
            out_ = torch.mm(torch.mm(torch.mm(rotation_x, rotation_y), rotation_z), trans)
            out_ = out_.contiguous().view(1, C, T, V)
            sub_out.append(out_)
        sub_out = torch.stack(sub_out).contiguous().view(N, C, T, V)
        sub_out = sub_out.permute(0, 2, 1, 3).contiguous().view(N, T, -1)
        out = self.main_lstm(sub_out)
        out = self.main_fc(out[:, :, -1].squeeze())

        t = out
        assert not ((t != t).any())

        return out
