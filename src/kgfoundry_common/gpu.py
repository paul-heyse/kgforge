"""Utilities for detecting GPU stack availability."""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Iterable

GPU_CORE_MODULES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "torchaudio",
    "vllm",
    "faiss",
    "triton",
    "cuvs",
    "cuda",
    "cupy",
)
"""Canonical module names that make up the optional GPU stack."""


def _modules_available(modules: Iterable[str]) -> bool:
    """Return True when every module in ``modules`` can be resolved.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    modules : str
        Describe ``modules``.

    Returns
    -------
    bool
        Describe return value.
"""
    return all(importlib.util.find_spec(module) is not None for module in modules)


def has_gpu_stack(*, allow_without_cuda_env: str = "ALLOW_GPU_TESTS_WITHOUT_CUDA") -> bool:
    """Return True when the optional GPU stack is importable and CUDA is usable.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    allow_without_cuda_env : str, optional
        Environment variable that, when set to ``"1"``, permits returning True even when
        CUDA is unavailable. This supports import-only validation on CPU-only hosts.
        Defaults to ``'ALLOW_GPU_TESTS_WITHOUT_CUDA'``.

    Returns
    -------
    bool
        True when the GPU stack is available or the override environment variable is set.
"""
    if not _modules_available(GPU_CORE_MODULES):
        return False
    if os.getenv(allow_without_cuda_env) == "1":
        return True
    try:
        torch_module = importlib.import_module("torch")
    except Exception:  # pragma: no cover - import guard
        return False
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None:
        return False
    is_available = getattr(cuda_module, "is_available", None)
    if not callable(is_available):
        return False
    return bool(is_available())
