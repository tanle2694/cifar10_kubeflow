import os
import abc
import logging
import json
import errno
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from cifarlibs.training.modeling.models.resnet_models import ResNet18
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from cifarlibs.training.modeling.metrics import accuracy
from cifarlibs.training.modeling.losses import cross_entropy
from cifarlibs.training.trainer_utils import MetricTracker, MonitorEarlyStop
from cifarlibs.utils.timer import Timer

logger = logging.getLogger(__name__)
torch.manual_seed(43)


class AbstractEvaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def start_evaluate(self):
        pass


class CifarEvaluator(AbstractEvaluator):
    def __init__(self, input_data_dir, batch_size, model, loss_name='cross_entropy'):
        self._input_data = input_data_dir
        self._batch_size = batch_size
        self._model = model
        self._loss_name = loss_name

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _dataset_setup(self):

        transform_test = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        test_dataset = torchvision.datasets.CIFAR10(root=self._input_data, download=False, train=False,
                                                     transform=transform_test)

        return test_dataset

    def _dataloader_setup(self):
        test_dataset = self._dataset_setup()
        self._test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self._batch_size, num_workers=4, pin_memory=True
        )
        return self._test_loader

    def _create_model(self):
        logger.info("Create model")
        net = ResNet18()
        net = net.to(self._device)
        if self._device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        net.load_state_dict(torch.load(self._model)['state_dict'])
        net.eval()
        self._net = net

    def _create_optim_ops(self):
        if self._loss_name == "cross_entropy":
            self.criterion = cross_entropy
        else:
            raise ValueError("Not support loss function")

    def start_evaluate(self):
        test_loss = 0
        correct = 0
        total = 0

        self._dataloader_setup()
        self._create_optim_ops()
        self._create_model()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self._test_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._net(inputs)
                loss = self.criterion(outputs, targets).item()
                test_loss += loss * targets.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 1. * correct / total
        avg_loss = 1. * test_loss / total
        return avg_loss, acc

    @staticmethod
    def save_result(avg_loss, avg_acc, file_output):
        if not os.path.exists(os.path.dirname(file_output)):
            try:
                os.makedirs(os.path.dirname(file_output))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(file_output, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"loss": avg_loss, "acc": avg_acc}))

