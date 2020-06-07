import os
from abc import abstractmethod
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import ensure_dir


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, device, device_ids, epochs, save_folder, monitor):
        self.device = device
        self.model = model.to(device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = epochs
        self.start_epoch = 1

        self.logs_folder = f"log/{datetime.now()}"
        self.checkpoint_location = os.path.join(save_folder, f"models/{model._get_name()}/{datetime.now()}/)")
        ensure_dir(self.logs_folder)
        ensure_dir(self.checkpoint_location)
        self.writer = SummaryWriter(os.path.join(save_folder, self.logs_folder))

        monitor_parts = monitor.split(maxsplit=1)
        self.monitor_mode = monitor_parts[0].lower()
        self.monitor_best = monitor_parts[1]
        self.monitor_early_stopping = 30

        assert (self.monitor_mode == "min" or self.monitor_mode == "max")

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        best = None
        early_stop_counter = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)

            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))

            is_better = best is None or self.monitor_mode == "min" and log[self.monitor_best] < best or \
                        self.monitor_mode == "max" and log[self.monitor_best] > best
            if is_better:
                early_stop_counter = 0
                best = log[self.monitor_best]
                path = f"{self.checkpoint_location}best_model.pth"
                torch.save(self.model.state_dict(), path)
                print("Found best model! Saved to:", path)
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.monitor_early_stopping:
                    print(f"Early stopping. {self.monitor_best} metric of best model: {best}")
                    return

            checkpoint = f"{self.checkpoint_location}epoch{epoch:03d}_valAcc{log[self.monitor_best]:.5f}.pth"
            torch.save(self.model.state_dict(), checkpoint)
            print("Saved to checkpoint:", checkpoint)
