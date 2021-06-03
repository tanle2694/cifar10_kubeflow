import pandas as pd

from numpy import inf
from matplotlib import pyplot as plt
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MetricTracker:
    def __init__(self, *keys, writer=None):
        # self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] * 1.0 / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class MonitorEarlyStop:
    def __init__(self, monitor_spec, nb_earlystop):
        self.mnt_variable, self.mnt_mode = monitor_spec.split("_")
        assert self.mnt_variable in ['acc', 'loss'], f"{self.mnt_variable} not support"
        assert self.mnt_mode in ['min', 'max'], f"{self.mnt_mode} not support"
        self.mnt_best = inf if self.mnt_mode == "min" else -inf
        self.nb_earlystop = nb_earlystop
        self.nb_not_improve = 0

    def check_to_stop(self, mnt_result):
        step_result = mnt_result[self.mnt_variable]
        improved = (self.mnt_mode == 'min' and step_result < self.mnt_best) or \
                   (self.mnt_mode == 'max' and step_result > self.mnt_best)

        if not improved:
            self.nb_not_improve += 1
            logger.info(f"Model not improved {self.mnt_variable} in {self.nb_not_improve} epochs ")
        else:
            logger.info(f"{self.mnt_variable} improve from {self.mnt_best} to {step_result}")

            self.mnt_best = step_result
            self.nb_not_improve = 0

        need_stop = False
        if self.nb_not_improve >= self.nb_earlystop:
            logger.info("Stop training because EarlyStop")
            need_stop = True
        return improved, need_stop

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))