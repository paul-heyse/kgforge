"""Type stubs for libcuvs library.

libcuvs provides CUDA-accelerated vector search operations used by FAISS
when GPU support is enabled. This stub module provides minimal type hints
for the library loading function.
"""

from __future__ import annotations

__all__ = ["load_library"]

def load_library(*args: object, **kwargs: object) -> None:
    """Load the libcuvs shared library.

    This function loads the CUDA-accelerated vector search library.
    It should be called before using FAISS with GPU support.

    Parameters
    ----------
    *args : object
        Variable positional arguments (unused).
    **kwargs : object
        Variable keyword arguments (unused).

    Notes
    -----
    - Safe to call multiple times (idempotent)
    - Raises ImportError if libcuvs is not available
    - Typically called before importing faiss in GPU-enabled code paths
    """
    ...
