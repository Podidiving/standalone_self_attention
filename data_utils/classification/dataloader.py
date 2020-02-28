from argparse import Namespace

from torch.utils.data import DataLoader

from .dataset import ClassificationDataset
from .augmentations import create_train_augmentor
from .augmentations import create_test_augmentor

from .dataset import image_read
from .dataset import image_read_one_channel


def create_train_dataloader(configs: Namespace):
    """
    :param configs: must have next properties:
        in_csv_train - path to csv
        root_prefix_train - root prefix for images
        image_path_column
        target_column
        image_size
        batch_size_train
        shuffle_train
        one_channel
        TODO rest params for dataloader
    :return: dataloader
    """
    image_read_func = image_read_one_channel if configs.one_channel else image_read
    transforms = create_train_augmentor(configs.image_size, configs.one_channel)
    dataset = ClassificationDataset(
        configs.in_csv_train,
        configs.root_prefix_train,
        configs.image_path_column,
        configs.target_column,
        transforms,
        image_read_func=image_read_func
    )
    return DataLoader(dataset, configs.batch_size_train, shuffle=configs.shuffle_train)


def create_test_dataloader(configs: Namespace):
    """
    :param configs: must have next properties:
        in_csv_test - path to csv
        root_prefix_test - root prefix for images
        image_path_column
        target_column
        image_size
        batch_size_test
        shuffle_test
        one_channel
        TODO rest params for dataloader
    :return: dataloader
    """
    transforms = create_test_augmentor(configs.image_size, configs.one_channel)
    image_read_func = image_read_one_channel if configs.one_channel else image_read
    dataset = ClassificationDataset(
        configs.in_csv_test,
        configs.root_prefix_test,
        configs.image_path_column,
        configs.target_column,
        transforms,
        image_read_func=image_read_func
    )
    return DataLoader(
        dataset, configs.batch_size_test,
        shuffle=configs.shuffle_test
    )
