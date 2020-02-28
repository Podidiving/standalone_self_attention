from typing import List, Union

import torch.nn as nn
import torch.nn.functional as F


from .encoder import ModelEncoder


class ClassificationModel(nn.Module):
    """
    ResNet Encoder + average pooling + dense layer
    Note: Because of fixed size of dense, you should pass
    only (32, 32) or (224, 224) images
    """
    def __init__(
            self,
            num_blocks: List[int],
            block_name: Union[str, List[str]],
            stem: bool = False,
            stem_spatial_downsample: str = 'soft',
            in_places: int = 64,
            expansion: int = 4,
            num_classes: int = 1000,
            one_channel: bool = False
    ):
        super().__init__()
        self.encoder = ModelEncoder(
            num_blocks,
            block_name,
            stem,
            stem_spatial_downsample,
            in_places,
            expansion,
            one_channel
        )
        self.dense = nn.Linear(512 * expansion, num_classes)

    def forward(self, x):
        out = self.encoder(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        return out


def resnet26_plain(
        stem_spatial_downsample: str = 'soft',
        num_classes: int = 1000,
        in_places: int = 64,
        expansion: int = 4,
        one_channel: bool = False
):
    return ClassificationModel(
        [1, 2, 4, 1],
        'resnet',
        stem=False,
        stem_spatial_downsample=stem_spatial_downsample,
        in_places=in_places,
        expansion=expansion,
        num_classes=num_classes,
        one_channel=one_channel
    )


def resnet26_attention(
        stem=False,
        stem_spatial_downsample: str = 'soft',
        num_classes: int = 1000,
        in_places: int = 64,
        expansion: int = 4,
        one_channel: bool = False
):
    return ClassificationModel(
        [1, 2, 4, 1],
        'attention_7_1_3_8_F',
        stem=stem,
        stem_spatial_downsample=stem_spatial_downsample,
        in_places=in_places,
        expansion=expansion,
        num_classes=num_classes,
        one_channel=one_channel
    )


def resnet38_plain(
        stem_spatial_downsample: str = 'soft',
        num_classes: int = 1000,
        in_places: int = 64,
        expansion: int = 4,
        one_channel: bool = False
):
    return ClassificationModel(
        [2, 3, 5, 2],
        'resnet',
        stem=False,
        stem_spatial_downsample=stem_spatial_downsample,
        in_places=in_places,
        expansion=expansion,
        num_classes=num_classes,
        one_channel=one_channel
    )


def resnet38_attention(
        stem=False,
        stem_spatial_downsample: str = 'soft',
        num_classes: int = 1000,
        in_places: int = 64,
        expansion: int = 4,
        one_channel: bool = False
):
    return ClassificationModel(
        [2, 3, 5, 2],
        'attention_7_1_3_8_F',
        stem=stem,
        stem_spatial_downsample=stem_spatial_downsample,
        in_places=in_places,
        expansion=expansion,
        num_classes=num_classes,
        one_channel=one_channel
    )


def resnet50_plain(
        stem_spatial_downsample: str = 'soft',
        num_classes: int = 1000,
        in_places: int = 64,
        expansion: int = 4,
        one_channel: bool = False
):
    return ClassificationModel(
        [3, 4, 6, 3],
        'resnet',
        stem=False,
        stem_spatial_downsample=stem_spatial_downsample,
        in_places=in_places,
        expansion=expansion,
        num_classes=num_classes,
        one_channel=one_channel
    )


def resnet50_attention(
        stem=False,
        stem_spatial_downsample: str = 'soft',
        num_classes: int = 1000,
        in_places: int = 64,
        expansion: int = 4,
        one_channel: bool = False
):
    return ClassificationModel(
        [3, 4, 6, 3],
        'attention_7_1_3_8_F',
        stem=stem,
        stem_spatial_downsample=stem_spatial_downsample,
        in_places=in_places,
        expansion=expansion,
        num_classes=num_classes,
        one_channel=one_channel
    )
