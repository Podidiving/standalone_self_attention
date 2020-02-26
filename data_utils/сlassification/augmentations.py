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
        albu.ImageCompression(quality_lower=80),
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


class ImageAugmentor:
    def __init__(
            self,
            transforms: albu.Compose
    ):
        self.transforms = transforms

    def __call__(self, image):
        return self.transforms(image=image)['image']


def create_train_augmentor(image_size=224):
    transforms = compose([
        pre_transforms(image_size=image_size),
        hard_transforms(),
        post_transforms()
    ])
    return ImageAugmentor(transforms)


def create_test_augmentor(image_size=224):
    transforms = compose([
        pre_transforms(image_size=image_size),
        post_transforms()
    ])
    return ImageAugmentor(transforms)
