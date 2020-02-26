from argparse import Namespace
from tqdm import tqdm
import numpy as np
import os

import torch

from visdom import Visdom

import models
# If you want to customize your train augmentations
# Consider changing
# data_utils.classifications.augmentations.create_train_augmentor
# function
from data_utils.сlassification import create_train_dataloader

from trainer.trainer_utils import get_criterion
from trainer.trainer_utils import get_optimizer
from trainer.trainer_utils import AverageMeter, ROCMeter
from trainer.trainer_utils import get_lr_scheduler


def create_model_from_configs(configs_: Namespace):
    def _create_model_name():
        return f'resnet{configs_.resnet_num}_{configs_.resnet_type}'

    if configs_.block_name is not None:
        assert type(configs_.block_name) is str or \
               (type(configs_.block_name) is list and
                configs_.num_blocks is not None and
                type(configs_.num_blocks) is list and
                len(configs_.block_name) == len(configs_.num_blocks))

        return models.ClassificationModel(
            configs_.num_blocks,
            configs_.block_name,
            configs_.stem,
            configs_.stem_spatial_downsample,
            configs_.in_places,
            configs_.expansion,
            configs_.num_classes
        )

    name = _create_model_name()
    try:
        model = models.__dict__[name](
            stem_spatial_downsample=configs_.stem_spatial_downsample,
            num_classes=configs_.num_classes,
            in_places=configs_.in_places,
            expansion=configs_.expansion
        )
    except KeyError:
        raise Exception(
            f'{name} is invalid model name'
        )
    return model


# Note that Trainer won't freeze your layers
# Cause we train models from scratch
class Trainer:
    def __init__(
            self,
            configs: Namespace
    ):
        self.configs = configs
        # model & training stuff
        self.device = self.configs.device
        self.model = create_model_from_configs(self.configs)
        # no multi gpu
        self.model.to(self.device)
        self.optimizer = get_optimizer(self.model, self.configs)
        self.criterion = get_criterion(self.configs).to(self.device)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.configs)
        # data
        self.dataloader_train = create_train_dataloader(self.configs)
        self.dataloader_test = None
        # utils
        self.state = None
        self.epoch = None
        self.training = None
        self.best_epoch = None

        # metrics
        self.train_loss = AverageMeter()
        self.test_loss = AverageMeter()
        self.test_metrics = ROCMeter()
        self.best_test_loss = AverageMeter()
        self.best_test_loss.update(np.array([np.inf]))

        if self.configs.visdom_log_file.endswit('.log'):
            self.visdom_log_file = self.configs.visdom_log_file
        else:
            self.visdom_log_file = os.path.join(
                self.configs.visdom_log_file,
                'visdom.log'
            )
        self.vis = Visdom(
            port=self.configs.visdom_port,
            log_to_filename=self.visdom_log_file,
            env=self.configs.visdom_env_name
        )

        self.vis_loss_opts = {
            'xlabel': 'epoch',
            'ylabel': 'loss',
            'title': 'losses',
            'legend': ['train_loss', 'test_loss']
        }

        self.vis_tpr_opts = {
            'xlabel': 'epoch',
            'ylabel': 'tpr',
            'title': 'val_tpr',
            'legend': ['tpr@fpr10-2', 'tpr@fpr10-3', 'tpr@fpr10-4']
        }

        self.vis_epochloss_opts = {
            'xlabel': 'epoch',
            'ylabel': 'loss',
            'title': 'epoch_losses',
            'legend': ['train_loss', 'val_loss']
        }

    def log(self, func):
        def wrapped(msg):
            if self.configs.log_file:
                with open(self.configs.log_file, 'a') as file:
                    _msg = msg if msg.endswith('\n') else msg + '\n'
                    file.write(_msg)
            func(msg)
        return wrapped

    def vislog_batch(self, batch_idx: int):
        if batch_idx % self.configs.log_batch_interval == 0:
            loader_len = len(self.dataloader_train) \
                if self.training else len(self.dataloader_test)
            cur_loss = self.train_loss if self.training else self.test_loss
            loss_type = 'train_loss' if self.training else 'val_loss'

            x_value = self.epoch + batch_idx / loader_len
            y_value = cur_loss.val
            self.vis.line([y_value], [x_value],
                          name=loss_type,
                          win='losses',
                          update='append')
            self.vis.update_window_opts(win='losses', opts=self.vis_loss_opts)

    def log_batch(self, batch_idx: int):
        if batch_idx % self.configs.log_batch_interval == 0:
            cur_len = len(self.dataloader_train) \
                if self.training else len(self.dataloader_test)
            cur_loss = self.train_loss if self.training else self.test_loss

            output_string = 'Train ' if self.training else 'Test '
            output_string += 'Epoch {}[{:.2f}%]:\t'.format(
                self.epoch,
                100. * batch_idx / cur_len
            )

            loss_i_string = 'Loss: {:.5f}({:.5f})\t'.format(cur_loss.val, cur_loss.avg)
            output_string += loss_i_string

            if not self.training:
                output_string += '\n'

                metrics_i_string = 'Accuracy: {:.5f}\t'.format(self.test_metrics.get_accuracy())
                output_string += metrics_i_string

            self.log(print)(output_string)
        self.vislog_batch(batch_idx)

    def log_epoch(self):
        out_train = 'Train: '
        out_test = 'Test:  '
        loss_i_string = 'Loss: {:.5f}\t'.format(self.train_loss.avg)
        out_train += loss_i_string
        loss_i_string = 'Loss: {:.5f}\t'.format(self.test_loss.avg)
        out_test += loss_i_string

        out_test += '\nTest:  '
        metrics_i_string = 'TPR@FPR=10-2: {:.4f}\t'.format(self.test_metrics.get_tpr(0.01))
        metrics_i_string += 'TPR@FPR=10-3: {:.4f}\t'.format(self.test_metrics.get_tpr(0.001))
        metrics_i_string += 'TPR@FPR=10-4: {:.4f}\t'.format(self.test_metrics.get_tpr(0.0001))
        out_test += metrics_i_string

        is_best = 'Best ' if self.best_epoch else ''
        out_res = is_best + 'Epoch {} results:\n'.format(self.epoch) + out_train + '\n' + out_test + '\n'

        self.log(print)(out_res)

    def vislog_epoch(self):
        x_value = self.epoch
        self.vis.line([self.train_loss.avg], [x_value],
                      name='train_loss',
                      win='epoch_losses',
                      update='append')
        self.vis.line([self.test_loss.avg], [x_value],
                      name='val_loss',
                      win='epoch_losses',
                      update='append')
        self.vis.update_window_opts(win='epoch_losses', opts=self.vis_epochloss_opts)

        self.vis.line([self.test_metrics.get_tpr(0.01)], [x_value],
                      name='tpr@fpr10-2',
                      win='val_tpr',
                      update='append')
        self.vis.line([self.test_metrics.get_tpr(0.001)], [x_value],
                      name='tpr@fpr10-3',
                      win='val_tpr',
                      update='append')
        self.vis.line([self.test_metrics.get_tpr(0.0001)], [x_value],
                      name='tpr@fpr10-4',
                      win='val_tpr',
                      update='append')
        self.vis.update_window_opts(win='val_tpr', opts=self.vis_tpr_opts)

    def on_epoch_end(self):
        self.log_epoch()
        self.vislog_epoch()

    def on_batch_end_train(self, batch_idx: int, loss_: float):
        self.train_loss.update(loss_)
        self.log_batch(batch_idx)

    def on_batch_end_test(
            self,
            batch_idx: int,
            target_batch: np.array,
            pred_batch: np.array,
            loss_: float
    ):
        self.test_loss.update(loss_)
        self.test_metrics.update(target_batch, pred_batch)
        self.log_batch(batch_idx)

    def train_epoch(self):
        self.model.train()
        self.training = True
        torch.set_grad_enabled(self.training)

        self.train_loss.reset()

        for batch_idx, (data_batch, target_batch) in \
                enumerate(self.dataloader_train):

            data_batch = data_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            self.optimizer.zero_grad()

            pred_batch = self.model(data_batch)
            if self.configs.loss_type == 'bce':
                target_batch = target_batch.float()
                cur_loss = self.criterion(pred_batch.squeeze(), target_batch)
            else:
                cur_loss = self.criterion(pred_batch, target_batch)
            cur_loss.backward()
            self.optimizer.step()
            self.on_batch_end_train(batch_idx, cur_loss.detach().cpu().item())

    def test_epoch(self):
        self.training = False
        torch.set_grad_enabled(self.training)
        self.model.eval()
        self.test_loss.reset()
        self.test_metrics.reset()
        for batch_idx, (data_batch, target_batch) in \
                enumerate(self.dataloader_test):
            data_batch = data_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            pred_batch = self.model(data_batch)
            if self.configs.loss_type == 'bce':
                target_batch = target_batch.float()
                cur_loss = self.criterion(pred_batch.squeeze(), target_batch)
            else:
                cur_loss = self.criterion(pred_batch, target_batch)
            if self.configs.loss_type == 'cce' or\
                    self.configs.loss_type == 'focal_loss':
                pred_batch = torch.nn.functional.softmax(pred_batch, dim=1)
            elif self.configs.loss_type == 'bce':
                pred_batch = torch.sigmoid(pred_batch)

            self.on_batch_end_test(
                batch_idx,
                target_batch.cpu().numpy(),
                pred_batch.cpu().numpy(),
                cur_loss.detach().cpu().item()
            )

        self.best_epoch = self.test_loss.avg < self.best_test_loss.val
        if self.best_epoch:
            # self.best_test_loss.val is container for best loss,
            # n is not used in the calculation
            self.best_test_loss.update(self.test_loss.avg, n=0)

    def resume(self):
        if os.path.isfile(self.configs.resume):
            loaded = torch.load(self.configs.resume)
            try:
                self.model.load_state_dict(loaded['model_state_dict'])
                self.optimizer.load_state_dict(loaded['optimizer'])
            except KeyError:
                self.model.load_state_dict(loaded)
            self.log(print)(f"Loaded checkpoint from {self.configs.resume}")
        else:
            self.log(print)(f"No file found at {self.configs.resume}")
        if os.path.isfile(self.visdom_log_file):
            self.vis.replay_log(log_filename=self.visdom_log_file)

    def freeze_layers(self, epoch: int):
        assert type(self.configs.freeze) is list
        for freeze_dict in self.configs.freeze:
            epoch_until = list(freeze_dict.keys())[0]
            if epoch <= epoch_until:
                chosen_layers = freeze_dict[epoch_until]
                if type(chosen_layers) is int:
                    assert chosen_layers > 0
                    # no multi gpu
                    for param in list(self.model.parameters())[:-chosen_layers]:
                        param.requires_grad = False
                    for param in list(self.model.parameters())[-chosen_layers:]:
                        param.requires_grad = True
                elif type(chosen_layers) is list:
                    if chosen_layers[0] > 0:
                        chosen_layers = np.array(chosen_layers)
                        assert np.all(chosen_layers >= 0)
                        for param in self.model.parameters():
                            param.requires_grad = False
                        for param in \
                                np.array(
                                    list(self.model.parameters())
                                )[chosen_layers]:
                            param.requires_grad = True
                    else:
                        chosen_layers = np.array(chosen_layers)
                        assert np.all(chosen_layers < 0)
                        for param in self.model.parameters():
                            param.requires_grad = True
                        for param in \
                                np.array(
                                    list(self.model.parameters())
                                )[chosen_layers]:
                            param.requires_grad = False
                else:
                    raise Exception(
                        'wrong freeze parameter type'
                    )
                return

    def train(self):

        if self.configs.resume:
            self.resume()

        start_epoch = self.configs.start_epoch
        end_epoch = self.configs.end_epoch
        range_ = range(start_epoch, end_epoch)
        if self.configs.verbose:
            range_ = tqdm(range_)

        for epoch in range_:
            self.epoch = epoch
            if self.configs.freeze:
                self.freeze_layers(epoch)
            self.train_epoch()
            self.test_epoch()
            self.lr_scheduler.step()

            self.create_state()
            self.save_state()
            self.on_epoch_end()

    def trace(self):
        assert self.configs.trace is not None  # TODO

    def create_state(self):
        self.state = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }

    def save_state(self):
        if self.configs.log_checkpoint == 0:
            self.save_checkpoint('checkpoint.pth')
        else:
            if self.best_epoch:
                self.save_checkpoint('model_best.pth')
            elif (self.epoch + 1) % self.configs.log_checkpoint == 0:
                self.save_checkpoint(f'model_{self.epoch}.pth')

    def save_checkpoint(self, filename):
        if self.configs.checkpoint_out is None:
            self.log(print)("provide checkpoint_out key to save checkpoints")
        fin_path = os.path.join(self.configs.checkpoint_out, filename)
        torch.save(self.state, fin_path)
