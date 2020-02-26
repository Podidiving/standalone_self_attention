from argparse import Namespace

from torch.utils.data import DataLoader

from .dataset import ClassificationDataset
from .augmentations import create_train_augmentor
from .augmentations import create_test_augmentor


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
        TODO rest params for dataloader
    :return: dataloader
    """
    transforms = create_train_augmentor(configs.image_size)
    dataset = ClassificationDataset(
        configs.in_csv_train,
        configs.root_prefix_train,
        configs.image_path_column,
        configs.target_column,
        transforms
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
        TODO rest params for dataloader
    :return: dataloader
    """
    transforms = create_test_augmentor(configs.image_size)
    dataset = ClassificationDataset(
        configs.in_csv_test,
        configs.root_prefix_test,
        configs.image_path_column,
        configs.target_column,
        transforms
    )
    return DataLoader(dataset, configs.batch_size_test, shuffle=configs.shuffle_test)
