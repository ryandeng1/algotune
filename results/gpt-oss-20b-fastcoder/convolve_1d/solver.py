from typing import Any
import numpy as np

class Solver:
    """
    Faster convolution using NumPy's built‑in implementation.  
    This eliminates the heavy SciPy dependency and gives tight
    performance for 1‑D array convolution.
    """
    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        # np.convolve supports `full`, `same`, and `valid` modes,
        # matching the behaviour of scipy.signal.convolve for 1‑D data.
        mode = getattr(self, "mode", "full")
        return np.convolve(a, b, mode=mode)