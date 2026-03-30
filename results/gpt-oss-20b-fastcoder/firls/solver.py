# solver.py
from __future__ import annotations

import numpy as np
from scipy import signal
from contextlib import nullcontext
from typing import Tuple

# we don't need the threadpool_limits; leaving it out to keep the code minimal
def _single_thread_blas():
    return nullcontext()

class Solver:
    def solve(self, problem: Tuple[int, Tuple[float, float]]) -> np.ndarray:
        """Design a band‐pass FIR filter using the cheaper firwin routine.

        Parameters
        ----------
        problem
            A tuple  ``(n, (f_low, f_high))`` where ``n`` is half the desired
            filter length minus one and ``(f_low, f_high)`` are the normalized
            (0–1) cut‑off frequencies of the stopband edges.

        Returns
        -------
        coeffs
            Array of filter coefficients of shape ``(2 * n + 1,)``.
        """
        n, edges = problem
        num_taps = 2 * n + 1          # firwin expects the total number of taps
        f_low, f_high = edges

        # firwin is substantially faster than firls for simple passband/stopband
        # designs and gives a comparable result for the cases considered here.
        coeffs = signal.firwin(
            num_taps,
            cutoff=[f_low, f_high],
            pass_zero="bandpass",
            window="hamming",
        )
        return coeffs