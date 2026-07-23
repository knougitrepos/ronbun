from __future__ import annotations

from collections.abc import Callable
import importlib
from typing import Any

from research.embeddings.base import FRModelFamily, ModelSpec
from research.embeddings.pytorch.adapter import (
    OutputSelector,
    PyTorchFaceEmbeddingAdapter,
)
from research.embeddings.pytorch.models import (
    AdaFacePyTorchAdapter,
    ArcFacePyTorchAdapter,
    MagFacePyTorchAdapter,
)

ModuleFactory = Callable[[ModelSpec], Any]

_ADAPTER_TYPES: dict[FRModelFamily, type[PyTorchFaceEmbeddingAdapter]] = {
    "arcface": ArcFacePyTorchAdapter,
    "adaface": AdaFacePyTorchAdapter,
    "magface": MagFacePyTorchAdapter,
}
_MODULE_FACTORIES: dict[FRModelFamily, ModuleFactory] = {}


class ModelFactoryUnavailableError(RuntimeError):
    """Raised when no verified repository-specific factory was registered."""


def load_pytorch_module_factory(factory_path: str) -> ModuleFactory:
    """Resolve an explicit ``package.module:function`` checkpoint loader."""

    resolved = str(factory_path).strip()
    if resolved.count(":") != 1:
        raise ValueError(
            "module factory path must use 'package.module:function' syntax"
        )
    module_name, attribute_name = (
        component.strip() for component in resolved.split(":", maxsplit=1)
    )
    if not module_name or not attribute_name:
        raise ValueError(
            "module factory path must use 'package.module:function' syntax"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"module factory is not callable: {resolved}")
    return factory


def register_pytorch_module_factory(
    family: FRModelFamily,
    factory: ModuleFactory,
    *,
    replace: bool = False,
) -> None:
    if family not in _ADAPTER_TYPES:
        raise ValueError(f"unsupported FR model family: {family}")
    if not callable(factory):
        raise TypeError("factory must be callable")
    if family in _MODULE_FACTORIES and not replace:
        raise ValueError(f"a PyTorch module factory is already registered for {family}")
    _MODULE_FACTORIES[family] = factory


def unregister_pytorch_module_factory(family: FRModelFamily) -> None:
    _MODULE_FACTORIES.pop(family, None)


def create_pytorch_adapter(
    spec: ModelSpec,
    *,
    module: Any | None = None,
    device: str = "cpu",
    output_selector: OutputSelector | None = None,
) -> PyTorchFaceEmbeddingAdapter:
    """Wrap an injected module or use an explicitly registered factory.

    No default "official" checkpoint loader is bundled because the three
    upstream repositories use different architectures and state-dict formats.
    This prevents silently loading a checkpoint into an incompatible network.
    """

    resolved_module = module
    if resolved_module is None:
        factory = _MODULE_FACTORIES.get(spec.family)
        if factory is None:
            raise ModelFactoryUnavailableError(
                f"no verified PyTorch module factory is registered for "
                f"{spec.family}; inject a loaded torch.nn.Module or register a "
                "repository-specific checkpoint factory"
            )
        resolved_module = factory(spec)
    adapter_type = _ADAPTER_TYPES[spec.family]
    return adapter_type(
        resolved_module,
        spec,
        device=device,
        output_selector=output_selector,
    )


def create_pytorch_adapter_from_spec(
    spec: ModelSpec,
    *,
    device: str = "cpu",
    output_selector: OutputSelector | None = None,
) -> PyTorchFaceEmbeddingAdapter:
    """Create an adapter from the explicit factory recorded in ``ModelSpec``."""

    if spec.module_factory is None:
        raise ModelFactoryUnavailableError(
            "model spec does not record a repository-specific module_factory"
        )
    module_factory = load_pytorch_module_factory(spec.module_factory)
    return create_pytorch_adapter(
        spec,
        module=module_factory(spec),
        device=device,
        output_selector=output_selector,
    )
