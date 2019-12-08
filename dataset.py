# this functions is mainly copy from
# catalyst's classification tutorial
# https://github.com/catalyst-team/catalyst/blob/master/examples/notebooks/classification-tutorial.ipynb
from catalyst.dl.utils import get_loader
from catalyst.utils.dataset import create_dataset, create_dataframe, prepare_dataset_labeling
from catalyst.utils.pandas import map_dataframe
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose
from catalyst.utils.dataset import split_dataframe
from catalyst.utils import imread
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import collections
import albumentations as albu
from albumentations.pytorch import ToTensor
from catalyst.data.augmentor import Augmentor


def show_examples(data, transforms=None):
    def read_random_images(data, transforms=None) -> List[Tuple[str, np.ndarray]]:
        data_ = np.random.choice(data, size=4)
        result = []
        for d in data_:
            image = imread(d['filepath'])
            if transforms is not None:
                image = transforms(image=image)['image']
            result.append((d['class'], image))
        return result
    images = read_random_images(data, transforms)
    _indexes = [(i, j) for i in range(2) for j in range(2)]

    f, ax = plt.subplots(2, 2, figsize=(16, 16))
    for (i, j), (title, img) in zip(_indexes, images):
        ax[i, j].imshow(img)
        ax[i, j].set_title(title)
    f.tight_layout()


def prepare_data(root, seed):
    dataset = create_dataset(dirs=f"{root}/*", extension="*.png")
    df = create_dataframe(dataset, columns=["class", "filepath"])

    tag_to_label = prepare_dataset_labeling(df, "class")
    class_names = [name for name, id_ in sorted(tag_to_label.items(), key=lambda x: x[1])]

    df_with_labels = map_dataframe(
        df,
        tag_column="class",
        class_column="label",
        tag2class=tag_to_label,
        verbose=False
    )

    train_data, valid_data = split_dataframe(df_with_labels, test_size=0.2, random_state=seed)
    train_data, valid_data = train_data.to_dict('records'), valid_data.to_dict('records')

    return train_data, valid_data, class_names


def create_reader(root, num_classes):

    # ReaderCompose collects different Readers into one pipeline
    open_fn = ReaderCompose([

        # Reads images from the `datapath` folder
        # using the key `input_key =" filepath "` (here should be the filename)
        # and writes it to the output dictionary by `output_key="features"` key
        ImageReader(
            input_key="filepath",
            output_key="features",
            datapath=root
        ),

        # Reads a number from our dataframe
        # by the key `input_key =" label "` to np.long
        # and writes it to the output dictionary by `output_key="targets"` key
        ScalarReader(
            input_key="label",
            output_key="targets",
            default_value=-1,
            dtype=np.int64
        ),

        # Same as above, but with one encoding
        ScalarReader(
            input_key="label",
            output_key="targets_one_hot",
            default_value=-1,
            dtype=np.int64,
            one_hot_classes=num_classes
        )
    ])

    return open_fn


def get_loaders(
        train_data: 'pd.DataFrame',
        valid_data: 'pd.DataFrame',
        open_fn: 'Callable',
        train_transforms_fn,
        valid_transforms_fn,
        batch_size: int = 64,
        num_workers: int = 4,
        sampler=None
) -> collections.OrderedDict:
    train_loader = get_loader(
        train_data,
        open_fn=open_fn,
        dict_transform=train_transforms_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=sampler is None,  # shuffle data only if Sampler is not specified (PyTorch requirement)
        sampler=sampler,
        drop_last=True,
    )

    valid_loader = get_loader(
        valid_data,
        open_fn=open_fn,
        dict_transform=valid_transforms_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=None,
        drop_last=True,
    )

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders


BORDER_CONSTANT = 0
BORDER_REFLECT = 2


def pre_transforms(image_size=32):
    # Convert the image to a square of size image_size x image_size
    # (keeping aspect ratio)
    result = [
        albu.LongestMaxSize(max_size=image_size),
        albu.PadIfNeeded(image_size, image_size, border_mode=BORDER_CONSTANT)
    ]

    return result


def hard_transforms():
    result = [
        # Random shifts, stretches and turns with a 50% probability
        albu.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=BORDER_REFLECT,
            p=0.5
        ),
        albu.IAAPerspective(scale=(0.02, 0.05), p=0.3),
        # Random brightness / contrast with a 30% probability
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        # Random gamma changes with a 30% probability
        albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
        # Randomly changes the hue, saturation, and color value of the input image
        albu.HueSaturationValue(p=0.3),
        albu.JpegCompression(quality_lower=80),
    ]

    return result


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensor()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


def get_transforms():
    train_transforms = compose([
        pre_transforms(),
        hard_transforms(),
        post_transforms()
    ])
    valid_transforms = compose([pre_transforms(), post_transforms()])

    show_transforms = compose([pre_transforms(), hard_transforms()])

    # Takes an image from the input dictionary by the key `dict_key`
    # and performs `train_transforms` on it.
    train_data_transforms = Augmentor(
        dict_key="features",
        augment_fn=lambda x: train_transforms(image=x)["image"]
    )

    # Similarly for the validation part of the dataset.
    # we only perform squaring, normalization and ToTensor
    valid_data_transforms = Augmentor(
        dict_key="features",
        augment_fn=lambda x: valid_transforms(image=x)["image"]
    )
    return train_data_transforms, valid_data_transforms, show_transforms
