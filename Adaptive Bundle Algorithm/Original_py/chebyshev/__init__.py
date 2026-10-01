"""Weighted-Chebyshev algorithms for the adaptive-bundle experiments.

Imports are deliberately lazy.  On macOS, importing cyipopt/IPOPT before
PyTorch can load conflicting OpenMP runtimes in the same process; the
experiment entry point imports the objective backend first and then imports
``chebyshev.weighted``.
"""

__all__ = [
    "chebyshev_adaptive",
    "scalarized_gradient",
    "scalarized_gn",
    "solve_weighted_chebyshev_dual",
    "weighted_chebyshev_exact_gd",
]


def __getattr__(name):
    if name in __all__:
        from . import weighted

        return getattr(weighted, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
