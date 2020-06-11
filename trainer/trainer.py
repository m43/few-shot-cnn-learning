# -*- coding: utf-8 -*-

import numpy as np
import torch

from base.base_trainer import BaseTrainer
from utils.util import MetricTracker


class OmniglotTrainer(BaseTrainer):
    """
    Omniglot trainer class
    """

    def __init__(self, run_name, model, criterion, metric_ftns, metric_ftns_oneshot, optimizer, device, device_ids,
                 epochs, save_folder, monitor, early_stopping, train_loader, val_loader=None, val_oneshot_loader=None, test_loader=None,
                 train_oneshot_loader=None, start_epoch=1, lr_scheduler=None):
        super().__init__(run_name, model, criterion, metric_ftns, optimizer, device, device_ids, epochs, save_folder,
                         monitor, start_epoch, early_stopping)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_oneshot_loader = val_oneshot_loader
        self.test_loader = test_loader  # TODO remove this :-)
        self.train_oneshot_loader = train_oneshot_loader  # TODO remove this as well

        self.metric_ftns_oneshot = metric_ftns_oneshot

        self.lr_scheduler = lr_scheduler
        self.len_epoch = len(self.train_loader)
        self.log_step = int(np.sqrt(self.train_loader.batch_size))

        self.train_metrics = MetricTracker("train", 'loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker("val", 'loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_oneshot_metrics = MetricTracker("val", *[m.__name__ for m in self.metric_ftns_oneshot])
        self.test_oneshot_metrics = MetricTracker("test", *[m.__name__ for m in self.metric_ftns_oneshot])
        self.train_oneshot_metrics = MetricTracker("train oneshot", *[m.__name__ for m in self.metric_ftns_oneshot])

        # input = next(iter(self.train_loader))[0].to(self.device)
        # input = input.transpose(1, 0)
        # self.writer.add_graph(model, [input[0], input[1]])

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            data = data.transpose(1, 0)
            output = self.model(data[0], data[1])
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            step = (epoch - 1) * self.len_epoch + batch_idx
            self.train_metrics.update('loss', loss.item())
            self.writer.add_scalar(f"{self.train_metrics.get_name()} loss", loss.item(), step)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
                # self.writer.add_scalar(f"{self.train_metrics.get_name()} {met.__name__}", met(output, target), step)

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))
                # self.writer.add_image(f"{self.train_metrics.get_name()} input", make_grid(
                #     [make_grid(data.cpu()[0], nrow=8, normalize=True),
                #      make_grid(data.cpu()[1], nrow=8, normalize=True)], nrow=2, normalize=True), step)

        log = self.train_metrics.result()
        self.writer.add_scalar(f"{self.train_metrics.get_name()} epoch loss", log["loss"], epoch)
        for met in self.metric_ftns:
            self.writer.add_scalar(f"{self.train_metrics.get_name()} epoch {met.__name__}", log[met.__name__], epoch)

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, epoch, bins='auto')  # TODO could this be slow?

        if self.train_oneshot_loader is not None:
            train_oneshot_log = self._oneshot_epoch(epoch, self.train_oneshot_loader, self.train_oneshot_metrics)
            log.update(**{f"{self.train_oneshot_metrics.get_name()} {k}": v for k, v in train_oneshot_log.items()})

        if self.val_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**{f"{self.valid_metrics.get_name()} {k}": v for k, v in val_log.items()})

        if self.val_oneshot_loader is not None:
            val_oneshot_log = self._oneshot_epoch(epoch, self.val_oneshot_loader, self.valid_oneshot_metrics)
            log.update(**{f"{self.valid_oneshot_metrics.get_name()} {k}": v for k, v in val_oneshot_log.items()})

        if self.test_loader is not None:
            # TODO remove this :-)
            test_log = self._oneshot_epoch(epoch, self.test_loader, self.test_oneshot_metrics)
            log.update(**{f"{self.test_oneshot_metrics.get_name()} {k}": v for k, v in test_log.items()})

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
                loss = self.criterion(output, target)

                # step = (epoch - 1) * len(self.val_loader) + batch_idx
                # self.writer.add_scalar(f"{self.valid_metrics.get_name()} loss", loss.item(), step)
                self.valid_metrics.update('loss', loss.item())

                self.model.eval()
                output = self.model(data[0], data[1])
                for met in self.metric_ftns:
                    m = met(output, target)
                    # self.writer.add_scalar(f"{self.valid_metrics.get_name()} {met.__name__}", m, step)
                    self.valid_metrics.update(met.__name__, m)

        result = self.valid_metrics.result()
        self.writer.add_scalar(f"{self.valid_metrics.get_name()} epoch loss", result["loss"], epoch)
        for met in self.metric_ftns:
            self.writer.add_scalar(f"{self.valid_metrics.get_name()} epoch {met.__name__}", result[met.__name__], epoch)
        return result

    def _oneshot_epoch(self, epoch, loader, metrics):
        self.model.eval()
        metrics.reset()

        with torch.no_grad():
            for batch_idx, (data,) in enumerate(loader):

                for i, current_batch_data in enumerate(data):
                    # TODO actually, it does not make much sense to use batches in here, not in this way as i've modeled
                    #  it where I have to anyways unzip the pairs inside each batch. Consider remodeling oneshot dataloader

                    current_batch_data = current_batch_data.to(self.device)
                    second = current_batch_data[1]
                    for target, first_image in enumerate(current_batch_data[0]):  # TODO write without last for loop
                        # Now one shot (first_image vs 20 images):
                        # TODO something is very slow in here
                        # TODO reconsider to preprocess all shots in advance
                        target = torch.tensor([target]).to(self.device)
                        output = self.model(first_image.expand(second.shape), second)

                        for met in self.metric_ftns_oneshot:
                            m = met(output, target)
                            metrics.update(met.__name__, m)

                        # if target == 9 and i % 3 == 0:
                        #     pred = output.max(0)[1]
                        #     # correct += int(pred == target)
                        #     # total += 1
                        #     print(epoch, "output", output.tolist())
                        #     print("\tpred", pred)
                        #     print("\ttarget", target)
                        #
                        #     self.writer.add_image(
                        #         f'epoch:{epoch}_{metrics.get_name()}_target:{target.item()}_pred:{pred.item()}___{output.tolist()}',
                        #         OmniglotVisualizer.make_next_batch_grid(torch.cat(
                        #             [first_image.expand(second.shape).unsqueeze(0), second.unsqueeze(0)]).transpose(1,
                        #                                                                                             0).cpu()))

            result = metrics.result()
            for met in self.metric_ftns_oneshot:
                self.writer.add_scalar(f"{metrics.get_name()} {met.__name__}", result[met.__name__], epoch)
            return result

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_loader, 'n_samples'):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
