import pandas as pd
import numpy as np
import os

from typing import Union, List

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
            in_csv: Union[str, List[str]],
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
        if type(self.in_csv) is str:
            self.in_csv = [self.in_csv]
        self.root_prefix = root_prefix
        self.image_path_column = image_path_column
        self.target_column = target_column
        self.dfs = None
        self.total_len = None
        self.__read_dfs()
        self.transforms = transforms or (lambda x: x)
        self.image_read_func = image_read_func

    def __read_dfs(self):
        self.dfs = []
        self.total_len = 0
        for in_csv in self.in_csv:
            df = pd.read_csv(in_csv)
            self.dfs.append(df)
            self.total_len += len(df)

    def __len__(self):
        return self.total_len

    def __get_df_idx(self, idx: int):
        cur_len, prev_len = 0, 0
        for df_idx, df in enumerate(self.dfs):
            prev_len = cur_len
            cur_len += len(df)
            if idx < cur_len:
                return df_idx, idx - prev_len
        raise Exception('Not supposed to be')

    def __getitem__(self, idx):
        df_idx, idx = self.__get_df_idx(idx)
        image_rel_path = self.dfs[df_idx][self.image_path_column].iloc[idx]
        image_path = os.path.join(
            self.root_prefix, image_rel_path
        )
        target = self.dfs[df_idx][self.target_column].iloc[idx]
        image = self.image_read_func(image_path)
        image = self.transforms(image)

        return image, target
