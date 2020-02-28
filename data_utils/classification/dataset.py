import pandas as pd
import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset


def image_read(image_path: str):
    return np.array(Image.open(image_path))


def image_read_one_channel(image_path: str):
    return np.expand_dims(
        np.array(Image.open(image_path)),
        -1
    )


class ClassificationDataset(Dataset):
    def __init__(
            self,
            in_csv: str,
            root_prefix: str,
            image_path_column: str = 'path',
            target_column: str = 'target',
            transforms=None,
            image_read_func=image_read
    ):
        """

        :param in_csv: path to your csv file, which contains your data
        csv dataset must opens valid with pd.read_csv call
        :param root_prefix: path to images will be constructed with this prefix
        :param image_path_column: column name in your csv 
        with relative path to image
        :param target_column: column name in your csv with target name for your
        image
        :param transforms: data augmentations.
        It's functions, that return image after call transforms(image)
        :param image_read_func: function to read image
        """
        self.in_csv = in_csv
        self.root_prefix = root_prefix
        self.image_path_column = image_path_column
        self.target_column = target_column
        self.df = pd.read_csv(self.in_csv)
        self.transforms = transforms or (lambda x: x)
        self.image_read_func = image_read_func

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_rel_path = self.df[self.image_path_column].iloc[idx]
        image_path = os.path.join(
            self.root_prefix, image_rel_path
        )
        target = self.df[self.target_column].iloc[idx]
        image = self.image_read_func(image_path)
        image = self.transforms(image)

        return image, target
