from argparse import Namespace
import torch


class DummyLR:
    def step(self):
        pass

    @staticmethod
    def state_dict():
        return None


def get_lr_scheduler(optimizer: torch.optim, configs: Namespace):
    try:
        return torch.optim.lr_scheduler.__dict__[configs.lr_scheduler_name](
            optimizer,
            **configs.lr_scheduler_params
        )
    except KeyError:
        print('Wrong lr scheduler. Use without it')
        return DummyLR()
