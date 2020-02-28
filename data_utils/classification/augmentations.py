import albumentations as albu
from albumentations.pytorch import ToTensorV2 as ToTensor


BORDER_CONSTANT = 0
BORDER_REFLECT = 2


def pre_transforms(image_size=224):
    # Convert the image to a square of size image_size x image_size
    # (keeping aspect ratio)
    result = [
        albu.LongestMaxSize(max_size=image_size),
        albu.PadIfNeeded(image_size, image_size, border_mode=BORDER_CONSTANT)
    ]

    return result


def hard_transforms(
        one_channel: bool = False
):
    result = [
        albu.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=BORDER_REFLECT,
            p=0.5
        ),
        albu.IAAPerspective(scale=(0.02, 0.05), p=0.3),
        # Random gamma changes with a 30% probability
        albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
    ]
    if not one_channel:
        result = result + [
            albu.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            ),
            albu.ImageCompression(quality_lower=80),
            albu.HueSaturationValue(p=0.3),
        ]

    return result


def post_transforms(one_channel: bool = False):
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    mean = [0.485, 0.456, 0.406] if not one_channel else [0.688438]
    std = [0.229, 0.224, 0.225] if not one_channel else [0.367651]
    return [albu.Normalize(mean=mean, std=std), ToTensor()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


class ImageAugmentor:
    def __init__(
            self,
            transforms: albu.Compose
    ):
        self.transforms = transforms

    def __call__(self, image):
        return self.transforms(image=image)['image']


def create_train_augmentor(
        image_size: int = 224,
        one_channel: bool = False
):
    transforms = compose([
        pre_transforms(image_size=image_size),
        hard_transforms(one_channel),
        post_transforms(one_channel)
    ])
    return ImageAugmentor(transforms)


def create_test_augmentor(
        image_size: int = 224,
        one_channel: bool = False
):
    transforms = compose([
        pre_transforms(image_size=image_size),
        post_transforms(one_channel)
    ])
    return ImageAugmentor(transforms)
