# solver.py
import numpy as np
from scipy.fft import dst
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> NDArray:
        """
        Compute the N-dimensional DST Type II with the view that scipy.fft.dst
        is implemented in fast C/Fortran code and performs well for all
        supported array shapes.
        """
        # Perform the multidimensional DST type II over all axes
        return dst(problem, type=2, axis=None)
