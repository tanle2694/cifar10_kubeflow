import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from resnet_models import ResNet18
from torch.utils.data.sampler import SubsetRandomSampler
from utils import Timer


import os
import argparse
import numpy as np

torch.manual_seed(43)

parser = argparse.ArgumentParser(description="Pytorch CIFAR10 training")

parser.add_argument('--batch_size', type=int, default=32, help="batch size use when training")
parser.add_argument('--lr', type=float, default=0.01, help="init learningrate")
parser.add_argument("--use_random_crop", type=bool, default=True, help="use random crop augmentation")
parser.add_argument("--use_horizon_flip", type=bool, default=True, help="use random horizontal flip augmentation")
parser.add_argument("--input_data", type=str, default="/home/tan.le2/PycharmProjects/cifar10_kubeflow/pytorch-cifar/data", help="Directory include data")
parser.add_argument("--epoch", type=int, default=1)

args = parser.parse_args()

train_compose = []

if args.use_random_crop:
    train_compose.append(transforms.RandomCrop(32, padding=4))

if args.use_horizon_flip:
    train_compose.append(transforms.RandomHorizontalFlip())

train_compose.extend([transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_train = transforms.Compose(train_compose)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(root=args.input_data, download=False, train=True,
                                              transform=transform_train)

valid_dataset = torchvision.datasets.CIFAR10(root=args.input_data, download=False, train=True,
                                              transform=transform_test)

best_acc = 0
best_val_loss = 1000
valid_size = 0.1 # 10% datatraining
random_seed = 123
iteration = 0
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True,
    )

valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size * 2, sampler=valid_sampler,
        num_workers=4, pin_memory=True,
    )


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print("Creating model")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train(args, epoch, iteration):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print("-------------------------------------------------------------------------")
    print(f"Epoch: {epoch}/{args.epoch}")
    timer = Timer()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        timer.step()
        if batch_idx % 100 == 0:
            avg_loss = '%.3f' % (train_loss / (batch_idx + 1))
            acc = '%.3f' % (100.*correct/total)
            speed = '%.3f' % timer.get_speed()
            print(f"Batch-id: {batch_idx}/{len(train_loader)} | Loss: {avg_loss}, "
                  f" | Acc: {acc}  | Correct/total: {correct}/{total} | Speed: {speed} batch/s")
            timer = Timer()
            break
    return iteration


def validation(epoch):
    global best_acc, best_val_loss
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("Validation")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    val_loss = test_loss / (len(valid_loader) + 1)
    acc = 100.*correct/total
    print(f"Val Loss: {'%.3f' % (val_loss)} | Best val loss: {'%.3f' % (best_val_loss)} | "
          f"Acc: {'%.3f' % (acc)} | Best Acc: {best_acc}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
    if acc > best_acc:
        print('Saving best model')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for i in range(args.epoch):
    iteration = train(args, i, iteration)
    validation(i)
