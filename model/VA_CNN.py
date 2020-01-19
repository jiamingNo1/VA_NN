# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 10:40
# @Author  : jiamingNo1
# @Email   : jiaming19.huang@foxmail.com
# @File    : VA_CNN.py
# @Software: PyCharm
import math
import torch
import torch.nn as nn
import torchvision.models as models


class VACNN(nn.Module):
    '''
    Input shape should be (N,C,T,V)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints.
    '''

    def __init__(self):
        super(VACNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(7)
        self.fc = nn.Linear(6272, 6)
        self.resnet_layer = models.resnet50()
        self.weights_init()

    def forward(self, x, target=None):
        N, C, T, V = x.size()
        min_val, max_val = -4.765629, 5.187813

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        sub_out = []
        for n in range(N):
            rotation_x = torch.tensor(
                [[1, 0, 0],
                 [0, math.cos(out[n, 0].item()), math.sin(out[n, 0].item())],
                 [0, math.sin(-out[n, 0].item()), math.cos(out[n, 0].item())]]
            )
            rotation_y = torch.tensor(
                [[math.cos(out[n, 1].item()), 0, math.sin(-out[n, 1].item())],
                 [0, 1, 0],
                 [math.sin(out[n, 1].item()), 0, math.cos(out[n, 1].item())]]
            )
            rotation_z = torch.tensor(
                [[math.cos(out[n, 2].item()), math.sin(out[n, 2].item()), 0],
                 [math.sin(-out[n, 2].item()), math.cos(out[n, 2].item()), 0],
                 [0, 0, 1]]
            )

            rotation = torch.mm(torch.mm(rotation_z, rotation_y), rotation_x)
            rotation = rotation.to('cuda')
            part_1 = torch.mm(rotation, x[n, :, :, :].view(C, T * V))
            part_2 = torch.div(torch.mm(rotation, (min_val + out[n, 3:6].view(-1, 1).expand(-1, T * V))) - min_val,
                               max_val - min_val)

            out_ = torch.add(part_1, torch.mul(255, part_2))
            out_ = out_.contiguous().view(C, T, V)
            sub_out.append(out_)

        sub_out = torch.stack(sub_out).contiguous().view(N, C, T, V)
        out = self.resnet_layer(sub_out)

        t = out
        assert not ((t != t).any())

        return out

    def weights_init(self):
        for layer in [self.conv1, self.conv2]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                if 'bias' in name:
                    param.data.zero_()
        for layer in [self.bn1, self.bn2]:
            layer.weight.data.fill_(1)
            layer.bias.data.fill_(0)
            layer.momentum = 0.99
            layer.eps = 1e-3

        self.fc.bias.data.zero_()
        self.fc.weight.data.zero_()


if __name__ == '__main__':
    model = VACNN()
    resnet50 = torch.load('weights/resnet50.pth')
    model_dict = model.state_dict()
    resnet50 = {'resnet_layer.' + k: v for k, v in resnet50.items() if 'resnet_layer.' + k in model_dict}
    model_dict.update(resnet50)
    model.load_state_dict(model_dict)
    in_features = model.resnet_layer.fc.in_features
    model.resnet_layer.fc = nn.Linear(in_features, 60)
    print(model.state_dict())
