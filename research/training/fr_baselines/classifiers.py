"""Loss classifiers for MobileFaceNet baselines.

Implements ArcFace, AdaFace, and MagFace classification heads based on the
hyperparameters and formulation specified in UCFace (IEEE T-IFS 2024,
DOI: 10.1109/TIFS.2024.3426973):

- ArcFace: m = 0.6, s = 64.0 (Deng et al., CVPR 2019)
- AdaFace: m = 0.4, s = 64.0, h = 0.333, adaptive feature norm tracking (Kim et al., CVPR 2022)
- MagFace: l_a = 10.0, u_a = 110.0, l_m = 0.45, u_m = 0.8, lambda_g = 20.0, s = 64.0 (Meng et al., CVPR 2021)
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class ArcFaceClassifier(nn.Module):
    """Additive Angular Margin (ArcFace) classification head."""

    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 1000,
        scale: float = 64.0,
        margin: float = 0.6,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = float(scale)
        self.margin = float(margin)
        self.easy_margin = bool(easy_margin)

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ArcFace scaled cosine logits.

        Args:
            features: Embeddings tensor of shape (B, in_features).
            labels: Ground-truth class indices of shape (B,). If None, standard
                scaled cosine similarities are returned without margin penalties.
        """
        cosine = F.linear(
            F.normalize(features, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        )
        if labels is None:
            return cosine * self.scale

        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-7, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale


class AdaFaceClassifier(nn.Module):
    """Quality Adaptive Margin (AdaFace) classification head.

    Adapts the angular and additive margin dynamically based on the input feature norm.
    """

    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 1000,
        scale: float = 64.0,
        margin: float = 0.4,
        h: float = 0.333,
        t_alpha: float = 0.01,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = float(scale)
        self.margin = float(margin)
        self.h = float(h)
        self.t_alpha = float(t_alpha)
        self.eps = float(eps)

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(10.0))

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute AdaFace scaled cosine logits.

        Args:
            features: Unnormalized or normalized embeddings of shape (B, in_features).
            labels: Ground truth class labels of shape (B,).
        """
        norms = torch.norm(features, 2, dim=-1, keepdim=True)
        safe_norms = torch.clamp(norms, min=0.001, max=100.0).clone().detach()

        cosine = F.linear(
            F.normalize(features, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        )
        if labels is None:
            return cosine * self.scale

        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        if self.training:
            with torch.no_grad():
                mean = safe_norms.mean()
                self.batch_mean.copy_(
                    self.batch_mean * (1.0 - self.t_alpha) + mean * self.t_alpha
                )
                if safe_norms.size(0) > 1:
                    std = safe_norms.std()
                    if not torch.isnan(std) and not torch.isinf(std):
                        self.batch_std.copy_(
                            self.batch_std * (1.0 - self.t_alpha) + std * self.t_alpha
                        )

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clamp(margin_scaler, -1.0, 1.0)

        g_angle = -self.margin * margin_scaler
        g_add = self.margin * margin_scaler + self.margin

        target_cosine = cosine.gather(1, labels.view(-1, 1).long())
        target_sine = torch.sqrt(
            torch.clamp(1.0 - target_cosine * target_cosine, min=1e-7, max=1.0)
        )

        cos_g = torch.cos(g_angle)
        sin_g = torch.sin(g_angle)
        phi = target_cosine * cos_g - target_sine * sin_g - g_add

        th = torch.cos(math.pi - torch.abs(g_angle))
        phi = torch.where(target_cosine > th, phi, target_cosine - g_add)

        output = cosine.scatter(1, labels.view(-1, 1).long(), phi)
        return output * self.scale


class MagFaceClassifier(nn.Module):
    """Magnitude-Aware Margin (MagFace) classification head.

    Applies an adaptive angular margin m(a_i) based on feature magnitude a_i
    and calculates an auxiliary regularization loss g(a_i).
    """

    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 1000,
        scale: float = 64.0,
        l_a: float = 10.0,
        u_a: float = 110.0,
        l_m: float = 0.45,
        u_m: float = 0.8,
        lambda_g: float = 20.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = float(scale)
        self.l_a = float(l_a)
        self.u_a = float(u_a)
        self.l_m = float(l_m)
        self.u_m = float(u_m)
        self.lambda_g = float(lambda_g)

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MagFace scaled logits and auxiliary magnitude loss.

        Args:
            features: Embeddings tensor of shape (B, in_features).
            labels: Ground truth labels of shape (B,).

        Returns:
            Tuple of (scaled_logits, auxiliary_magnitude_loss).
        """
        norms = torch.norm(features, 2, dim=-1, keepdim=True)
        cosine = F.linear(
            F.normalize(features, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        )
        if labels is None:
            zero_aux = torch.tensor(0.0, device=features.device, dtype=features.dtype)
            return cosine * self.scale, zero_aux

        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        a = torch.clamp(norms, min=self.l_a, max=self.u_a)
        a_margin = a.detach()
        m = ((self.u_m - self.l_m) / (self.u_a - self.l_a)) * (a_margin - self.l_a) + self.l_m

        target_cosine = cosine.gather(1, labels.view(-1, 1).long())
        target_sine = torch.sqrt(
            torch.clamp(1.0 - target_cosine * target_cosine, min=1e-7, max=1.0)
        )

        cos_m = torch.cos(m)
        sin_m = torch.sin(m)
        phi = target_cosine * cos_m - target_sine * sin_m

        th = torch.cos(math.pi - m)
        mm = torch.sin(math.pi - m) * m
        phi = torch.where(target_cosine > th, phi, target_cosine - mm)

        output = cosine.scatter(1, labels.view(-1, 1).long(), phi) * self.scale

        # MagFace magnitude regularization loss: g(a) = 1/a + a / (u_a^2)
        g_mag = (1.0 / a) + (a / (self.u_a**2))
        aux_loss = self.lambda_g * torch.mean(g_mag)

        return output, aux_loss
