import os
import abc
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from cifarlibs.training.modeling.models.resnet_models import ResNet18
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from cifarlibs.training.modeling.metrics import accuracy
from cifarlibs.training.modeling.losses import cross_entropy
from cifarlibs.training.trainer_utils import MetricTracker

logger = logging.getLogger(__name__)
torch.manual_seed(43)


class AbstractTrainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def start_train(self):
        pass

    @abc.abstractmethod
    def wait_for_completion(self):
        pass

    @abc.abstractmethod
    def register_model(self):
        pass


class CifarTrainer(AbstractTrainer):
    def __init__(self, input_data, epochs, batch_size, lr, use_random_crop, use_horizon_flip, loss_name, weight_decay,
                 optimizer_algo, metrics_name):
        self.input_data = input_data
        self.batch_size = batch_size
        self.lr = lr
        self.use_random_crop = use_random_crop
        self.use_horizon_flip = use_horizon_flip
        self.validation_size = 0.1 # 10% datatraining
        self.random_seed = 123
        self.num_worker = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_name = loss_name
        self.weight_decay = weight_decay
        self.optimizer_algo = optimizer_algo
        self.epochs = epochs
        self.metrics_name = metrics_name
        self.train_metrics = MetricTracker('loss', *self.metrics_name)

    def dataset_setup(self):
        train_compose = []
        if self.use_random_crop:
            train_compose.append(transforms.RandomCrop(32, padding=4))

        if self.use_horizon_flip:
            train_compose.append(transforms.RandomHorizontalFlip())

        train_compose.extend([transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_train = transforms.Compose(train_compose)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.train_dataset = torchvision.datasets.CIFAR10(root=self.input_data, download=False, train=True,
                                                     transform=transform_train)

        self.valid_dataset = torchvision.datasets.CIFAR10(root=self.input_data, download=False, train=True,
                                                     transform=transform_test)
        return self.train_dataset, self.valid_dataset

    def dataloader_setup(self, train_dataset, valid_dataset):
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.validation_size * num_train))
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_worker, pin_memory=True,
        )

        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=self.batch_size * 2, sampler=valid_sampler,
            num_workers=self.num_worker, pin_memory=True,
        )

        return self.train_loader, self.valid_loader

    def create_model(self):
        net = ResNet18()
        net = net.to(self.device)
        if self.device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        self.net = net

    def create_optim_ops(self):
        if self.loss_name == "cross_entropy":
            self.criterion = cross_entropy
        else:
            raise ValueError("Not support loss function")

        if self.optimizer_algo == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
                              momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_algo == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.99),
                                        eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        else:
            raise ValueError("Not support optimzer algorithm")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def start_train(self):
        not_improved_count = 0
        for epoch in range(self.epochs):
            pass

    def _train_epoch(self):
        self.net.train()
        self.train_metrics.reset()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            batch_train_loss = loss.item()
            if 'loss' in self.metrics_name:
                self.train_metrics.update('loss', batch_train_loss)

            for met_name in self.metrics_name:
                if met_name == "loss":
                    continue
                try:
                    metric_func = getattr(metrics, met_name)
                except Exception:
                    raise ValueError("Metric not found")

                self.train_metrics.update(met_name, metric_func(outputs, targets))

