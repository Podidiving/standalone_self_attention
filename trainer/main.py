from argparse import ArgumentParser
from argparse import Namespace
import os

import safitty

from .classification_trainer.trainer import Trainer as ClfTrainer


def read_configs(conf_path):
    assert os.path.isfile(conf_path)
    assert conf_path.endswith('yml')
    return Namespace(**safitty.load(conf_path))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--type',
        type=str,
        required=False,
        default='classification',
        help='type of task to train model on'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    type_ = args.type
    configs = read_configs(args.config)
    if type_ == 'classification':
        trainer = ClfTrainer(configs)
        trainer.train()
        if configs.trace is not None:
            trainer.trace()
    else:
        raise NotImplementedError
