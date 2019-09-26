# -*-coding: utf-8*-
# @Time: 2019/8/26 下午7:24
# @Author: jiamingNo1
# FileName: main.py
# Software: PyCharm
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from datetime import datetime

from model.VA_CNN import VACNN
from model.VA_RNN import VARNN
from data.feeder import fetch_dataloader


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "y", "1")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='data/', help='root directory for all datasets')
parser.add_argument('--dataset_name', default='NTU-RGB+D-CV', help='dataset name')
parser.add_argument('--save_dir', default='results/', help='root directory for saving checkpoint models')
parser.add_argument('--log_dir', default='logs/', help='root directory for train and test log')
parser.add_argument('--model_name', default='VACNN', help='model name')
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument('--cuda', default='True', type=str2bool, help='use cuda to train model')


def main():
    # params
    args = parser.parse_args()
    json_file = 'config/params.json'
    with open(json_file) as f:
        params = json.load(f)
    params['dataset_dir'] = args.dataset_dir
    params['dataset_name'] = args.dataset_name

    # Training settings
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print("WARNING: It looks like you have a CUDA device,but aren't" +
                  "using CUDA. \nRun with --cuda for optimal training speed")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.model_name == 'VACNN':
        model = VACNN()
        model = nn.DataParallel(model).to(device)
    elif args.model_name == 'VARNN':
        model = VARNN()
        model = nn.DataParallel(model).to(device)
    else:
        raise ValueError()

    if not os.path.exists(args.save_dir + args.model_name):
        os.mkdir(args.save_dir + args.model_name)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    # optimizer mode
    if params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    elif params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['lr'])
    else:
        raise ValueError()

    # data loader and learning rate strategy
    train_loader = fetch_dataloader('train', params)
    val_loader = fetch_dataloader('val', params)
    test_loader = fetch_dataloader('test', params)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, cooldown=2, verbose=True)

    # tensorboard
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    # train and test
    if args.mode == 'train':
        writer = SummaryWriter('{}{}/'.format(args.log_dir, args.model_name) + time_stamp)
        for epoch in range(params['max_epoch']):
            train(writer, model, optimizer, device, train_loader, epoch)
            if (epoch + 1) % 1000 == 0:
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            },
                           args.save_dir + args.model_name + '/{}.pth'.format(str(epoch + 1)))
                print("{:%Y-%m-%dT%H-%M-%S} saved model {}".format(
                    datetime.now(),
                    args.save_dir + args.model_name + '/{}.pth'.format(str(epoch + 1))))
            current = val(writer, model, device, val_loader, epoch)
            lr_scheduler.step(current)
        print('Finished Training')
        writer.close()
    else:
        test(model, device, test_loader)


def train(writer, model, optimizer, device, train_loader, epoch):
    model.train()
    losses = 0.0
    acces = 0.0

    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        acces += pred.eq(target.view_as(pred)).sum().item()

        if (idx + 1) % 100 == 0:
            writer.add_scalar('Loss/Train',
                              losses / 100,
                              epoch * len(train_loader) + idx + 1)
            writer.add_scalar('Accuracy/Train',
                              acces / 3200,
                              epoch * len(train_loader) + idx + 1)
            print(
                "{:%Y-%m-%dT%H-%M-%S}  epoch:{}  batch:{}  (loss:{:.3f} acc:{:.3f}).".format(datetime.now(), epoch + 1,
                                                                                             idx + 1,
                                                                                             losses / 100,
                                                                                             acces / 3200))
            acces, losses = 0.0, 0.0


def val(writer, model, device, val_loader, epoch):
    model.eval()
    loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(val_loader.dataset)
        writer.add_scalar('Accuracy/Val',
                          100. * correct / len(val_loader.dataset),
                          epoch + 1)
        writer.add_scalar('Loss/Val',
                          loss,
                          epoch + 1)
        print('(Val Set)  Epoch:{}  Average Loss: {:.3f}, Accuracy: {}/{} ({:.3f}%)'.
              format(epoch + 1, loss, correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))

    return 100.0 * correct / len(val_loader.dataset)


def test(model, device, test_loader):
    model.eval()
    loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(test_loader.dataset)
        print('(Test Set) Average Loss: {:.3f}, Accuracy: {}/{} ({:.3f}%)'.
              format(loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
