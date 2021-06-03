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


class AbstractTrainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def start_train(self):
        pass
    #
    # @abc.abstractmethod
    # def wait_for_completion(self):
    #     pass
    #
    # @abc.abstractmethod
    # def register_model(self):
    #     pass


class CifarTrainer(AbstractTrainer):
    def __init__(self, input_data_dir, epochs, batch_size, lr, use_random_crop, use_horizon_flip, loss_name, weight_decay,
                 optimizer_algo, monitor_spec, nb_earlystop, checkpoint_path, tensorboard_path):
        self._input_data = input_data_dir
        self._batch_size = batch_size
        self._lr = lr
        self._use_random_crop = use_random_crop
        self._use_horizon_flip = use_horizon_flip
        self._validation_size = 0.1 # 10% datatraining
        self._random_seed = 123
        self._num_worker = 4
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._loss_name = loss_name
        self._weight_decay = weight_decay
        self._optimizer_algo = optimizer_algo
        self._epochs = epochs

        self._total_iters = 0

        self._ES_monitor = MonitorEarlyStop(monitor_spec, nb_earlystop)
        self._checkpoint_path = checkpoint_path
        self._tensorboard_path = tensorboard_path

    def _dataset_setup(self):
        train_compose = []
        if self._use_random_crop:
            train_compose.append(transforms.RandomCrop(32, padding=4))

        if self._use_horizon_flip:
            train_compose.append(transforms.RandomHorizontalFlip())

        train_compose.extend([transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_train = transforms.Compose(train_compose)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root=self._input_data, download=False, train=True,
                                                     transform=transform_train)

        valid_dataset = torchvision.datasets.CIFAR10(root=self._input_data, download=False, train=True,
                                                     transform=transform_test)
        return train_dataset, valid_dataset

    def _dataloader_setup(self):
        train_dataset, valid_dataset = self._dataset_setup()
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self._validation_size * num_train))
        np.random.seed(self._random_seed)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self._batch_size, sampler=train_sampler,
            num_workers=self._num_worker, pin_memory=True,
        )

        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=self._batch_size * 2, sampler=valid_sampler,
            num_workers=self._num_worker, pin_memory=True,
        )

        return self.train_loader, self.valid_loader

    def _create_model(self):
        net = ResNet18()
        net = net.to(self._device)
        if self._device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        self.net = net

    def _create_optim_ops(self):
        if self._loss_name == "cross_entropy":
            self.criterion = cross_entropy
        else:
            raise ValueError("Not support loss function")

        if self._optimizer_algo == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(), lr=self._lr,
                                       momentum=0.9, weight_decay=self._weight_decay)
        elif self._optimizer_algo == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self._lr, betas=(0.9, 0.99),
                                        eps=1e-08, weight_decay=self._weight_decay, amsgrad=False)
        else:
            raise ValueError("Not support optimzer algorithm")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def start_train(self):

        self._dataloader_setup()
        self._create_model()
        self._create_optim_ops()

        for epoch in range(self._epochs):
            self._train_epoch(epoch)
            need_stop = self._validation(epoch)
            if need_stop:
                break

    def _update_metric_trackers(self, loss, outputs, targets, batch_idx, timer, tracker):
        acc = accuracy(outputs, targets)
        current_batch_size = len(targets)
        tracker.update('loss', loss, current_batch_size )
        tracker.update('acc', acc, current_batch_size)

        if batch_idx % 100 == 0:
            training_speed = timer.get_speed()
            timer = Timer()

            avg_loss = '%.3f' % (tracker.avg('loss'))
            acc = '%.3f' % (tracker.avg('acc'))
            speed = '%.3f' % training_speed
            curr_lr = self.scheduler.get_lr()
            logger.info(f"Batch-id: {batch_idx}/{len(self.train_loader)} | Loss: {avg_loss}, "
                  f" | Acc: {acc} | Speed: {speed} batch/s | Lr: {curr_lr}")

        return tracker, timer

    def _train_epoch(self, epoch):
        logger.info(f"Epoch {epoch}")
        self.net.train()
        train_metrics = MetricTracker("acc", 'loss')
        timer = Timer()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            timer.step()
            self._total_iters += 1

            inputs, targets = inputs.to(self._device), targets.to(self._device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_train_loss = loss.item()
            tracker, timer = self._update_metric_trackers(batch_train_loss, outputs, targets, batch_idx,
                                                          timer, train_metrics)

    def _validation(self, epoch):
        logger.info("Start validation")
        self.net.eval()
        validation_metrics = MetricTracker('acc', 'loss')
        timer = Timer()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.valid_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets).item()
                validation_metrics, timer = self._update_metric_trackers(loss, outputs, targets, -1,
                                                                         timer, validation_metrics)
        timer.step()
        val_loss = validation_metrics.avg('loss')
        acc = validation_metrics.avg('acc')
        logger.info(f"Val Loss: {'%.3f' % (val_loss)} | Acc: {'%.3f' % (acc)}")
        logger.info(f"Time to validation: {timer.get_speed()}s")
        improved, need_stop = self._ES_monitor.check_to_stop({'acc': acc, "loss": val_loss})
        if improved:
            self._save_checkpoint()
        return need_stop

    def _save_checkpoint(self):
        state = {
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if not os.path.exists(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)
        torch.save(state, os.path.join(self._checkpoint_path, "model_best.pth"))
        logger.info("Saved best model")




