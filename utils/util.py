import importlib
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.sigmoid(output) > 0.5
        assert len(pred) == len(target)
        correct = 0
        correct += torch.sum(pred.squeeze().long() == target).item()
    return correct / len(target)


def accuracy_oneshot(output, target):
    with torch.no_grad():
        # TODO support batch
        pred = output.max(0)[1]
        assert len(pred) == len(target)
        correct = torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=0)[1]
        # assert pred.shape[0] == len(target)
        assert pred.shape[1] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[i] == target).item()
    return correct / len(target)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class TensorboardWriter():
    def __init__(self, log_dir, enabled=True):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                          "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                          "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                # logger.warning(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


if __name__ == '__main__':
    metric_ftns = [accuracy]
    metric_ftns_oneshot = [accuracy, top_k_acc]
    writer = None
    train_metrics = MetricTracker('loss', *[m.__name__ for m in metric_ftns], writer=writer)
    valid_metrics = MetricTracker('loss', *[m.__name__ for m in metric_ftns], writer=writer)
    valid_oneshot_metrics = MetricTracker('loss', *[m.__name__ for m in metric_ftns_oneshot], writer=writer)
    test_oneshot_metrics = MetricTracker('loss', *[m.__name__ for m in metric_ftns_oneshot], writer=writer)

    train_metrics.reset()
    valid_metrics.reset()
    valid_oneshot_metrics.reset()
    test_oneshot_metrics.reset()

    train_metrics.reset()
    print(train_metrics._data)
    train_metrics.update('loss', 123)
    print(train_metrics._data)
    train_metrics.update('loss', 1)
    train_metrics.update('loss', 2)
    train_metrics.update('loss', 3)
    train_metrics.update('accuracy', 0.9)
    print(train_metrics._data)
    train_metrics.result()
