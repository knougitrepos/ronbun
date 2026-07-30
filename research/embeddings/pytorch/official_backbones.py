"""PyTorch backbones compatible with the selected official FR repositories.

The layer names and tensor flow follow the upstream implementations used by:

- InsightFace ArcFace PyTorch (MIT):
  https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
- AdaFace (MIT):
  https://github.com/mk-minchul/AdaFace
- MagFace (Apache-2.0):
  https://github.com/IrvingMeng/MagFace

Only the inference backbone is implemented here. Training heads and losses are
outside the Step 2 research scope. AdaFace returns the pre-normalization 512D
tensor so the common adapter can preserve its feature norm before applying the
shared L2-normalization contract.
"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


class InsightIBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        *,
        stride: int = 1,
        downsample: nn.Module | None = None,
        eps: float,
        momentum: float,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=eps, momentum=momentum)
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, eps=eps, momentum=momentum)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes, eps=eps, momentum=momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs: Any) -> Any:
        identity = inputs
        output = self.bn1(inputs)
        output = self.conv1(output)
        output = self.bn2(output)
        output = self.prelu(output)
        output = self.conv2(output)
        output = self.bn3(output)
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return output + identity


class InsightIResNet(nn.Module):
    """IResNet layout shared by official ArcFace and MagFace checkpoints."""

    fc_scale = 7 * 7

    def __init__(
        self,
        layers: Sequence[int],
        *,
        embedding_dim: int = 512,
        eps: float,
        momentum: float,
        dropout: float,
        dropout_2d: bool,
        freeze_output_scale: bool,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self._eps = eps
        self._momentum = momentum
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64, eps=eps, momentum=momentum)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=eps, momentum=momentum)
        dropout_type = nn.Dropout2d if dropout_2d else nn.Dropout
        self.dropout = dropout_type(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * self.fc_scale, embedding_dim)
        self.features = nn.BatchNorm1d(
            embedding_dim, eps=eps, momentum=momentum
        )
        if freeze_output_scale:
            nn.init.constant_(self.features.weight, 1.0)
            self.features.weight.requires_grad = False

    def _make_layer(self, planes: int, blocks: int, *, stride: int) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    planes, eps=self._eps, momentum=self._momentum
                ),
            )
        layers: list[nn.Module] = [
            InsightIBasicBlock(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                eps=self._eps,
                momentum=self._momentum,
            )
        ]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                InsightIBasicBlock(
                    self.inplanes,
                    planes,
                    eps=self._eps,
                    momentum=self._momentum,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, inputs: Any) -> Any:
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.prelu(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.bn2(output)
        output = self.dropout(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return self.features(output)


_INSIGHT_LAYERS = {
    "iresnet18": (2, 2, 2, 2),
    "iresnet34": (3, 4, 6, 3),
    "iresnet50": (3, 4, 14, 3),
    "iresnet100": (3, 13, 30, 3),
    "iresnet200": (6, 26, 60, 6),
}


def build_arcface_backbone(
    architecture: str, *, embedding_dim: int = 512
) -> InsightIResNet:
    try:
        layers = _INSIGHT_LAYERS[architecture]
    except KeyError as exc:
        raise ValueError(
            f"unsupported official ArcFace architecture: {architecture!r}"
        ) from exc
    return InsightIResNet(
        layers,
        embedding_dim=embedding_dim,
        eps=1e-5,
        momentum=0.1,
        dropout=0.0,
        dropout_2d=False,
        freeze_output_scale=True,
    )


def build_magface_backbone(
    architecture: str, *, embedding_dim: int = 512
) -> InsightIResNet:
    supported = {
        key: value
        for key, value in _INSIGHT_LAYERS.items()
        if key != "iresnet200"
    }
    try:
        layers = supported[architecture]
    except KeyError as exc:
        raise ValueError(
            f"unsupported official MagFace architecture: {architecture!r}"
        ) from exc
    return InsightIResNet(
        layers,
        embedding_dim=embedding_dim,
        eps=2e-5,
        momentum=0.9,
        dropout=0.4,
        dropout_2d=True,
        freeze_output_scale=False,
    )


class AdaFlatten(nn.Module):
    def forward(self, inputs: Any) -> Any:
        return inputs.view(inputs.size(0), -1)


class AdaSEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Any) -> Any:
        scale = self.avg_pool(inputs)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return inputs * self.sigmoid(scale)


class AdaBasicBlockIR(nn.Module):
    def __init__(
        self,
        in_channel: int,
        depth: int,
        stride: int,
        *,
        use_se: bool,
    ) -> None:
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(
                    in_channel, depth, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(depth),
            )
        residual: list[nn.Module] = [
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(
                in_channel, depth, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(
                depth, depth, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(depth),
        ]
        if use_se:
            residual.append(AdaSEModule(depth))
        self.res_layer = nn.Sequential(*residual)

    def forward(self, inputs: Any) -> Any:
        return self.res_layer(inputs) + self.shortcut_layer(inputs)


AdaBottleneck = namedtuple(
    "AdaBottleneck", ["in_channel", "depth", "stride"]
)


def _ada_block(
    in_channel: int, depth: int, units: int
) -> list[AdaBottleneck]:
    return [AdaBottleneck(in_channel, depth, 2)] + [
        AdaBottleneck(depth, depth, 1) for _ in range(units - 1)
    ]


_ADAFACE_LAYOUTS = {
    "ir_18": (2, 2, 2, 2, False),
    "ir_34": (3, 4, 6, 3, False),
    "ir_50": (3, 4, 14, 3, False),
    "ir_101": (3, 13, 30, 3, False),
    "ir_se_50": (3, 4, 14, 3, True),
}


class AdaFaceBackbone(nn.Module):
    """AdaFace IR backbone returning the raw feature before L2 normalization."""

    def __init__(
        self,
        architecture: str,
        *,
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        try:
            unit_counts = _ADAFACE_LAYOUTS[architecture]
        except KeyError as exc:
            raise ValueError(
                f"unsupported official AdaFace architecture: {architecture!r}"
            ) from exc
        counts = unit_counts[:4]
        use_se = bool(unit_counts[4])
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        blocks: list[AdaBottleneck] = []
        for in_channel, depth, units in zip(
            (64, 64, 128, 256),
            (64, 128, 256, 512),
            counts,
        ):
            blocks.extend(_ada_block(in_channel, depth, int(units)))
        self.body = nn.Sequential(
            *[
                AdaBasicBlockIR(
                    block.in_channel,
                    block.depth,
                    block.stride,
                    use_se=use_se,
                )
                for block in blocks
            ]
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            AdaFlatten(),
            nn.Linear(512 * 7 * 7, embedding_dim),
            nn.BatchNorm1d(embedding_dim, affine=False),
        )

    def forward(self, inputs: Any) -> Any:
        output = self.input_layer(inputs)
        output = self.body(output)
        return self.output_layer(output)


def build_adaface_backbone(
    architecture: str, *, embedding_dim: int = 512
) -> AdaFaceBackbone:
    return AdaFaceBackbone(architecture, embedding_dim=embedding_dim)
