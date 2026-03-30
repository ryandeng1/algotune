#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
solver.py

Highly optimised 2D convolution implementation for the 'full' mode with
a 'fill' (zero) boundary. For small arrays the direct algorithm from
SciPy is fast, but for larger arrays an FFT‑based approach outperforms
all other methods.  No external packages beyond NumPy are required.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from typing import Tuple


class Solver:
    """
    Computes the 2‑D convolution of two input arrays using the optimal
    algorithm for the array size.

    Parameters
    ----------
    threshold : int, optional (default: 64)
        The number of elements above which the FFT convolution is used.
        For very small arrays a direct implementation is faster.
    """

    def __init__(self, threshold: int = 64) -> None:
        self.boundary: str = "fill"
        self.mode: str = "full"
        self.threshold = threshold

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def __call__(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Alias for ``solve``."""
        return self.solve(problem)

    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the 2D convolution of arrays a and b using "full" mode
        and "fill" boundary.

        Parameters
        ----------
        problem: Tuple[np.ndarray, np.ndarray]
            Tuple containing the two 2‑D input arrays.
        Returns
        -------
        np.ndarray
            Array containing the convolution result.
        """
        a, b = problem
        # Quick exit for empty inputs
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0] + b.shape[0] - 1,
                             a.shape[1] + b.shape[1] - 1), dtype=a.dtype)

        # For very small arrays the direct SciPy implementation is faster
        if a.size * b.size < self.threshold ** 2:
            return signal.convolve2d(a, b, mode=self.mode, boundary=self.boundary)

        # -------------------------------
        # FFT‑based convolution
        # -------------------------------
        out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)

        # Pad to optimal size (next power of two may speed FFTs on some platforms)
        fft_shape = [self._next_pow2(sz) for sz in out_shape]

        # Zero‑pad inputs
        A = np.fft.rfftn(a, fft_shape)
        B = np.fft.rfftn(b, fft_shape)

        # Element‑wise product in frequency domain
        R = A * B

        # Inverse FFT to spatial domain
        conv = np.fft.irfftn(R, fft_shape)

        # Slice to the exact output size
        result = conv[: out_shape[0], : out_shape[1]]

        # Numerical errors may produce tiny imaginary parts; discard them
        if np.iscomplexobj(result):
            result = np.real(result)
        return result

    # ------------------------------------------------------------------ #
    # Utility functions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _next_pow2(x: int) -> int:
        """Return the next power of two greater than or equal to x."""
        return 1 << (x - 1).bit_length()