"""CUDA production backend for the discretize-first MCEF equations.

GPU symbols are imported lazily so the archive-only ``diagnose`` command can
run on login/analysis nodes that do not expose ``libcuda.so``.
"""

from importlib import import_module

__all__ = (
    "discrete_rhs_gpu",
    "full_step_discrete_bh",
    "make_discrete_gpu_model",
)


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.gpu_core"), name)
