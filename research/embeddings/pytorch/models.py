from __future__ import annotations

from research.embeddings.pytorch.adapter import PyTorchFaceEmbeddingAdapter


class ArcFacePyTorchAdapter(PyTorchFaceEmbeddingAdapter):
    family = "arcface"


class AdaFacePyTorchAdapter(PyTorchFaceEmbeddingAdapter):
    family = "adaface"


class MagFacePyTorchAdapter(PyTorchFaceEmbeddingAdapter):
    family = "magface"
