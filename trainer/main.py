from argparse import ArgumentParser
from argparse import Namespace
from typing import List
import os

import safitty

from trainer.classification_trainer.trainer import Trainer as ClfTrainer


def read_configs(conf_paths: List[str]) -> Namespace:
    assert type(conf_paths) is list
    configs_ = {}
    for conf_path in conf_paths:
        assert os.path.isfile(conf_path)
        assert conf_path.endswith('yml')
        cur_configs = safitty.load(conf_path)
        configs_.update(cur_configs)
    return Namespace(**configs_)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--configs',
        nargs='+',
        required=True,
        help='Paths to configs file'
    )
    parser.add_argument(
        '--type',
        type=str,
        required=False,
        default='classification',
        help='type of task to train model on'
    )
    return parser.parse_args()


def makedirs_if_needed(configs_: Namespace):
    if configs_.checkpoint_out is not None:
        os.makedirs(configs_.checkpoint_out, exist_ok=True)
    if configs.log_file is not None:
        dirs = configs.log_file.rsplit('/', 1)[0]
        os.makedirs(dirs, exist_ok=True)
    if configs_.visdom_log_file is not None:
        if configs_.visdom_log_file.endswith('.log'):
            dirs = configs_.visdom_log_file.rsplit('/', 1)[0]
            os.makedirs(dirs, exist_ok=True)
            with open(configs_.visdom_log_file, 'w') as _:
                pass
        else:
            os.makedirs(configs_.visdom_log_file, exist_ok=True)


if __name__ == '__main__':
    args = parse_args()
    type_ = args.type
    configs = read_configs(args.configs)
    makedirs_if_needed(configs)
    if type_ == 'classification':
        print('Classification Trainer is about to get started')
        trainer = ClfTrainer(configs)
        trainer.train()
        if configs.trace is not None:
            trainer.trace()
    else:
        raise NotImplementedError
