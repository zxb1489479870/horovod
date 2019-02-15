#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
        help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
        help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
        help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
        help='SGD momentum (default: 0.5)')
args = parser.parse_args()

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)


def partition_train_dataset():
    """ Partitioning MNIST train"""
    dataset = datasets.MNIST(
        'data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz =int(args.batch_size / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz

def partition_test_dataset():
    """ Partitioning MNIST test """
    dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz =int(args.test_batch_size / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    test_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return test_set, bsz

def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

torch.manual_seed(1234)
model = Net()

def train(rank, size):
    """ Distributed Synchronous SGD Example """
    model.train()
    train_set, train_bsz = partition_train_dataset()
    #model = model.cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #print(len(train_set.dataset))
    num_batches = ceil(len(train_set.dataset) / float(train_bsz))
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
            #data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ': ',
              epoch_loss / num_batches)

def test(rank, size):
    # run test
    model.eval()
    #model = model.cuda(rank)
    test_set, test_bsz = partition_test_dataset()
    num_batches = ceil(len(test_set.dataset) / float(test_bsz))
    test_loss = 0
    test_accuracy = 0.
    #print(len(test_set.dataset))
    for data, target in test_set:
        data, target = Variable(data), Variable(target)
#       data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
        output = model(data)
        loss = F.nll_loss(output, target)
        #test_loss += loss.item()
        test_loss += loss
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    test_loss /= num_batches
    test_accuracy /= len(test_set.dataset)
    dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(test_accuracy, op=dist.ReduceOp.SUM)
    #print(dist.get_world_size())
    if (dist.get_rank() == 0):
        print('Average loss for test set is : ',
                test_loss.item() / dist.get_world_size())
       # print(len(test_set.dataset))
        print('Average accuracy is : ',
                100. * test_accuracy.item() / dist.get_world_size())

def run(rank, size):
    train(rank, size)
    test(rank, size)

def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    init_processes(0, 0, run, backend='mpi')
    '''
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    '''
