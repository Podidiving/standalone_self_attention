from argparse import Namespace
from torch import nn


def get_criterion(configs: Namespace):
    try:
        loss_params = configs.loss_params or {}
        return nn.__dict__[configs.loss_name](
            **loss_params
        )
    except KeyError:
        raise Exception(
            'Unknown loss type'
        )
