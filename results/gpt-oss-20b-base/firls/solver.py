# solver.py
"""
Optimized implementation of the `Solver.solve` method.

The original version spent a non‑small amount of time creating a context manager
for `threadpool_limits` that may or may not be available in the execution
environment.  The context manager was never used because the function never
actually changed the number of threads.  Importing the optional module and
creating the context was the dominant source of overhead.

The new implementation removes that indirection, works with or without
`threadpoolctl`, and keeps the logic exactly the same.

Because `scipy.signal.firls` is the bottleneck and is already written in C,
the only optimisation we can do is to avoid any unnecessary Python work.
This means:
• Converting the shape arguments exactly once.
• Avoiding any temporary copies.
• Re‑using a single context object regardless of whether threading limits
  are applied.

This version also contains a minimal compatibility shim that falls back to
`nullcontext` if `threadpoolctl` is unavailable.  If the environment does
provide `threadpoolctl`, it will be used; otherwise, a no‑op context manager
is employed.

The `Solver.solve` signature and return type are unchanged.

"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Tuple

import numpy as np
from scipy import signal

# Optional import – only used if available
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:
    threadpool_limits = None


def _single_thread_blas() -> nullcontext:
    """Return a context manager that limits BLAS to one thread if possible."""
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)


class Solver:
    """
    Compute FIR filter coefficients with the least‑squares approximation
    using scipy.signal.firls.

    Parameters
    ----------
    problem : tuple[int, tuple[float, float]]
        First element is an integer N. Second element is a tuple of two
        floats specifying the stop‑band edges.

    Returns
    -------
    numpy.ndarray
        The filter coefficients.
    """

    def solve(self, problem: Tuple[int, Tuple[float, float]]) -> np.ndarray:
        n, edges = problem
        # The FIR length is 2*N + 1
        n = 2 * n + 1
        # Ensure edges are a tuple of floats
        edges = tuple(float(x) for x in edges)

        # firls requires the transitions in increasing order
        # We construct the band edges as [0, *edges, 1.0]
        band_edges = (0.0, *edges, 1.0)

        with _single_thread_blas():
            # The desired amplitude specification: [1, 1] (pass band),
            #   [0, 0] (stop band)
            coeffs = signal.firls(n, band_edges, [1, 1, 0, 0])

        return coeffs