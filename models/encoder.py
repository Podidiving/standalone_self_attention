from typing import List, Union

import torch.nn as nn


from models.utils.bottleneck import BottleNeck


class ModelEncoder(nn.Module):
    """
    Implements logic of resnet encoder class
    """
    def __init__(
            self,
            num_blocks: List[int],
            block_name: Union[str, List[str]],
            stem: bool = False,
            stem_spatial_downsample: str = 'soft',
            in_places: int = 64,
            expansion: int = 4,
            one_channel: bool = False
    ):
        super().__init__()
        assert stem_spatial_downsample in ['soft', 'hard'], \
            f"stem_spatial_downsample must be in ['soft', 'hard']" + \
            f" (soft for small input size, like CIFAR and hard for" + \
            f" big input size, like IMAGENET"
        self.in_places = in_places
        self.expansion = expansion
        self.first_channels = 1 if one_channel else 3
        if stem:
            raise NotImplementedError
        else:
            if stem_spatial_downsample == 'soft':
                self.init = nn.Sequential(
                    nn.Conv2d(
                        self.first_channels, 64, kernel_size=3,
                        stride=1, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            else:
                self.init = nn.Sequential(
                    nn.Conv2d(
                        self.first_channels, 64, kernel_size=7,
                        stride=2, padding=3, bias=False
                    ),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
        if type(block_name) is list:
            assert len(block_name) == len(num_blocks)
        else:
            block_name = [block_name] * len(num_blocks)
        self.block_name = block_name
        self.num_blocks = num_blocks
        self.layer1 = self._make_layer(64, 0, stride=1)
        self.layer2 = self._make_layer(128, 1, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, planes, idx, stride):
        num_blocks = self.num_blocks[idx]
        block_name = self.block_name[idx]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                BottleNeck(
                    self.in_places,
                    planes,
                    block_name,
                    stride=stride,
                    expansion=self.expansion
                )
            )
            self.in_places = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out
