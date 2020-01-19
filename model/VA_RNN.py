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
    Input shape should be (N,T,S)
    where N is the number of samples,
          T is the length of the sequence, 300
          V*C is the number of joints and coordinates, 50*3.
    '''

    def __init__(self, n_hid=100):
        super(VARNN, self).__init__()
        self.rotation_lstm = nn.LSTM(150, n_hid, batch_first=True)
        self.rotation_fc = nn.Linear(n_hid, 3)
        self.translation_lstm = nn.LSTM(150, n_hid, batch_first=True)
        self.translation_fc = nn.Linear(n_hid, 3)
        self.main_lstm = nn.LSTM(150, n_hid, 3, batch_first=True, dropout=0.5)
        self.main_fc = nn.Linear(n_hid, 60)
        self.dropout = nn.Dropout(0.5)
        self.weights_init()

    def forward(self, x, target=None):
        N, T, _ = x.size()

        rotation, _ = self.rotation_lstm(x)
        rotation = self.dropout(rotation)
        rotation = self.rotation_fc(rotation)
        translation, _ = self.translation_lstm(x)
        translation = self.dropout(translation)
        translation = self.translation_fc(translation)

        n_out = []
        for n in range(N):
            t_out = []
            for t in range(T):
                rotation_x = torch.tensor(
                    [[1, 0, 0],
                     [0, math.cos(rotation[n, t, 0].item()), math.sin(rotation[n, t, 0].item())],
                     [0, math.sin(-rotation[n, t, 0].item()), math.cos(rotation[n, t, 0].item())]])
                rotation_y = torch.tensor(
                    [[math.cos(rotation[n, t, 1].item()), 0, math.sin(-rotation[n, t, 1].item())],
                     [0, 1, 0],
                     [math.sin(rotation[n, t, 1].item()), 0, math.cos(rotation[n, t, 1].item())]])
                rotation_z = torch.tensor(
                    [[math.cos(rotation[n, t, 2].item()), math.sin(rotation[n, t, 2].item()), 0],
                     [math.sin(-rotation[n, t, 2].item()), math.cos(rotation[n, t, 2].item()), 0],
                     [0, 0, 1]])
                trans = x[n, t, :].view(-1, 3).permute(1, 0) - translation[n, t, :].view(3, -1).expand(3, 50)
                rotation_xyz = torch.mm(torch.mm(rotation_z, rotation_y), rotation_x)
                rotation_xyz = rotation_xyz.to('cuda')
                out_ = torch.mm(rotation_xyz, trans)
                t_out.append(out_)

            t_out = torch.stack(t_out)
            n_out.append(t_out)
        n_out = torch.stack(n_out)
        n_out = n_out.permute(0, 1, 3, 2).contiguous().view(N, T, -1)
        out, _ = self.main_lstm(n_out)
        out = self.dropout(out)
        out = self.main_fc(out)
        out = nn.Softmax(dim=1)(out.mean(1))

        return out

    def weights_init(self):
        for layer in [self.rotation_lstm, self.translation_lstm, self.main_lstm]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=0.001)

        self.rotation_fc.weight.data.zero_()
        self.translation_fc.weight.data.zero_()
        self.rotation_fc.bias.data.zero_()
        self.translation_fc.bias.data.zero_()


if __name__ == '__main__':
    model = VARNN()
    print(model.state_dict())
