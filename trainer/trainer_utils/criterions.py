from argparse import Namespace
from torch import nn


def get_criterion(configs: Namespace):
    try:
        return nn.__dict__[configs.loss_type](
            configs.loss_params
        )
    except KeyError:
        raise Exception(
            'Unknown loss type'
        )
