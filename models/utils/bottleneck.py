import re

from torch import nn
import torch.nn.functional as F


from models.self_attention_block import SelfAttentionBlock


class BottleNeck(nn.Module):
    """
    It's resnet bottle neck class
    You can pass custom block, which will be used
    between downsampling and upsampling
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            block_name: str,
            stride: int = 1,
            groups: int = 1,
            base_width: int = 64,
            expansion: int = 4
    ):
        super().__init__()
        self.expansion = expansion
        self.stride = stride
        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        if block_name == 'resnet':
            block = nn.Conv2d(
                width, width, kernel_size=3, padding=1
            )
        elif block_name.startswith('attention'):
            x = re.match(r"attention_(\d)_(\d)_(\d)_(\d)_(\w)", block_name)
            assert x is not None
            k_size_ = int(x.groups()[0])
            stride_ = int(x.groups()[1])
            pad_ = int(x.groups()[2])
            g_size_ = int(x.groups()[3])
            bias_ = True if x.groups()[4] == 'T' else False
            block = SelfAttentionBlock(
                width,
                width,
                kernel_size=k_size_,
                stride=stride_,
                padding=pad_,
                groups=g_size_,
                bias=bias_
            )
        else:
            raise Exception(
                f'Unknown block_name : {block_name}'
            )
        self.conv2 = nn.Sequential(
            block,
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, self.expansion * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride >= 2:
            out = F.avg_pool2d(out, (self.stride, self.stride))

        out += self.shortcut(x)
        out = F.relu(out)

        return out
