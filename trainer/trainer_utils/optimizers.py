from argparse import Namespace
from torch import nn
from torch import optim

from catalyst.contrib.nn import optimizers


def get_optimizer(
        model: nn.Module,
        configs: Namespace,
):
    """
    Create & compile optimizer
    TODO is it the best way to pass the whole model in this function?
    :param model:
    :param configs:
    :return:
    """
    try:
        optimizer_params = configs.optimizer_params or {}
        return optim.__dict__[configs.optimizer_name](
            model.parameters(),
            **optimizer_params
        )
    except KeyError:
        try:
            optimizer_params = configs.optimizer_params or {}
            return optimizers.__dict__[configs.optimizer_name](
                model.parameters(),
                **optimizer_params
            )
        except KeyError:
            raise Exception(
                'Unknown optimizer type'
            )
