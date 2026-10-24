"""FaceX-Zoo MobileFaceNet inference backbone, with upstream state-dict names.

Adapted from ``backbone/MobileFaceNets.py`` (Jun Wang, 2020-10-19), originally
based on https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/model.py.
Original author contact: jun21wangustc@gmail.com.
Upstream: https://github.com/JDAI-CV/FaceX-Zoo
Audited source SHA-256:
3691d5c90f8f23c1f02d0e8e5d8add37b88f21b1a2558b3d7d6b9ea382a74593

Copyright [2020] JD.com, Inc., JD AI
Licensed under the Apache License, Version 2.0. You may obtain a copy at
http://www.apache.org/licenses/LICENSE-2.0 . Distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for permissions and limitations.
The upstream license and bundled attribution notices are in LICENSE.facex-zoo.

Changes: typed, consolidated inference-only implementation; fixed 112x112
input and 7x7 depthwise output kernel; dropout omitted because inference uses
eval mode. All upstream parameter/buffer names and forward operations remain.
The training head is deliberately excluded: all four head families share the
same backbone forward and preserve its raw pre-normalization feature vector.
"""

from __future__ import annotations

from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class _LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class _DepthWise(nn.Module):
    def __init__(self, in_c, out_c, *, groups, stride=2, residual=False):
        super().__init__()
        self.conv = _ConvBlock(in_c, groups)
        self.conv_dw = _ConvBlock(groups, groups, kernel=3, stride=stride, padding=1, groups=groups)
        self.project = _LinearBlock(groups, out_c)
        self.residual = residual

    def forward(self, x):
        out = self.project(self.conv_dw(self.conv(x)))
        return x + out if self.residual else out


class _Residual(nn.Module):
    def __init__(self, channels, blocks, groups):
        super().__init__()
        self.model = nn.Sequential(*(
            _DepthWise(channels, channels, groups=groups, stride=1, residual=True)
            for _ in range(blocks)
        ))

    def forward(self, x):
        return self.model(x)


class FaceXMobileFaceNet(nn.Module):
    """Raw 512D or 1024D features for already prepared 112x112 BGR inputs."""

    def __init__(self, embedding_dim: int = 1024):
        super().__init__()
        if embedding_dim not in (512, 1024):
            raise ValueError("FaceX MobileFaceNet supports 512D or 1024D")
        self.conv1 = _ConvBlock(3, 64, kernel=3, stride=2, padding=1)
        self.conv2_dw = _ConvBlock(64, 64, kernel=3, padding=1, groups=64)
        self.conv_23 = _DepthWise(64, 64, groups=128)
        self.conv_3 = _Residual(64, 4, 128)
        self.conv_34 = _DepthWise(64, 128, groups=256)
        self.conv_4 = _Residual(128, 6, 256)
        self.conv_45 = _DepthWise(128, 128, groups=512)
        self.conv_5 = _Residual(128, 2, 256)
        self.conv_6_sep = _ConvBlock(128, 512)
        self.conv_6_dw = _LinearBlock(512, 512, kernel=7, groups=512)
        self.conv_6_flatten = nn.Flatten()
        self.linear = nn.Linear(512, embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.conv2_dw(self.conv1(x))
        x = self.conv_3(self.conv_23(x))
        x = self.conv_4(self.conv_34(x))
        x = self.conv_5(self.conv_45(x))
        x = self.conv_6_flatten(self.conv_6_dw(self.conv_6_sep(x)))
        return self.bn(self.linear(x))
