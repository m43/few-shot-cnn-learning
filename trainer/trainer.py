# -*- coding: utf-8 -*-

import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer
from data_loader import OmniglotVisualizer
from utils.util import MetricTracker


class OmniglotTrainer(BaseTrainer):
    """
    Omniglot trainer class
    """

    def __init__(self, model, criterion, metric_ftns, metric_ftns_oneshot, optimizer, device, device_ids, epochs,
                 writer, monitor, train_loader, val_loader=None, val_oneshot_loader=None, test_loader=None,
                 lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, device, device_ids, epochs, writer, monitor)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_oneshot_loader = val_oneshot_loader
        self.test_loader = test_loader  # TODO remove this :-)

        self.metric_ftns_oneshot = metric_ftns_oneshot

        self.lr_scheduler = lr_scheduler
        self.len_epoch = len(self.train_loader)
        self.log_step = int(np.sqrt(self.train_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_oneshot_metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns_oneshot], writer=self.writer)
        self.test_oneshot_metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns_oneshot], writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            data = data.transpose(1, 0)
            output = self.model(data[0], data[1])
            loss = self.criterion(output, target.unsqueeze(1).float())
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))
                self.writer.add_image('input', make_grid([make_grid(data.cpu()[0], nrow=8, normalize=True),
                                                          make_grid(data.cpu()[1], nrow=8, normalize=True)], nrow=2,
                                                         normalize=True))

            # if epoch > 1:
            #     with torch.no_grad():
            #         pred = torch.sigmoid(output) > 0.5
            #         assert pred.shape[0] == len(target)
            #         correct = 0
            #         correct += torch.sum(pred.squeeze().long() == target).item()
            #     print("Target", target)
            #     print("Predic", pred.squeeze().long())
            #     print(correct / len(target))

        log = self.train_metrics.result()

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        if self.val_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.val_oneshot_loader is not None:
            prefix = "val_"
            val_oneshot_log = self._oneshot_epoch(epoch, self.val_oneshot_loader, self.valid_oneshot_metrics, prefix)
            log.update(**{prefix + k: v for k, v in val_oneshot_log.items()})

        if self.test_loader is not None:
            # TODO remove this :-)
            prefix = "test_"
            test_log = self._oneshot_epoch(epoch, self.test_loader, self.test_oneshot_metrics, prefix)
            log.update(**{prefix + k: v for k, v in test_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()  # TODO
        return log

    def _valid_epoch(self, epoch):
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                self.model.train()
                data, target = data.to(self.device), target.to(self.device)

                data = data.transpose(1, 0)
                output = self.model(data[0], data[1])
                loss = self.criterion(output, target.unsqueeze(1).float())

                self.writer.set_step((epoch - 1) * len(self.val_loader) + batch_idx, 'valid')  # TODO check this
                self.valid_metrics.update('loss', loss.item())

                self.model.eval()
                output = self.model(data[0], data[1])
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', OmniglotVisualizer.make_next_batch_grid(data.cpu()))

        return self.valid_metrics.result()

    def _oneshot_epoch(self, epoch, loader, metrics, name="valid"):

        self.model.eval()
        self.valid_metrics.reset()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data,) in enumerate(loader):
                for current_batch_data in data:
                    # TODO actually, it does not make much sense to use batches in here, not in this way as i've modeled
                    #  it where I have to anyways unzip the pairs inside each batch. Consider remodeling oneshot dataloader
                    self.writer.set_step((epoch - 1) * len(loader) * 20 + batch_idx * 20, name)  # TODO check this

                    current_batch_data = current_batch_data.to(self.device)
                    second = current_batch_data[1]
                    for target, first_image in enumerate(current_batch_data[0]):  # TODO write without last for loop
                        # Now one shot (first_image vs 20 images):
                        target = torch.tensor([target]).to(self.device)
                        output = self.model(first_image.expand(second.shape), second)

                        pred = output.max(0)[1]
                        correct += int(pred == target)
                        total += 1

                        for met in self.metric_ftns_oneshot:
                            metrics.update(met.__name__, met(output, target))
                        self.writer.add_image('input',
                                              OmniglotVisualizer.make_next_oneshot_batch_grid(current_batch_data.cpu()))

            print(correct/total)
            return metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_loader, 'n_samples'):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
