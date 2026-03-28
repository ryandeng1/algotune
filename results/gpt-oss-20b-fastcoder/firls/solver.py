import numpy as np
from scipy import signal

def solve(problem: tuple[int, tuple[float, float]]) -> np.ndarray:
    """
    Construct a linear-phase FIR filter using the Parks–McClellan algorithm
    (scipy.signal.firls).  The input `problem` consists of an integer `n`
    and a tuple `(low, high)` defining the pass‑band edges.  The filter
    order is taken to be `2*n+1` as required by the original implementation.

    Parameters
    ----------
    problem : tuple[int, tuple[float, float]]
        (n, (low, high))

    Returns
    -------
    np.ndarray
        Array of filter coefficients.
    """
    n, edges = problem
    # Desired filter order (number of taps) is 2*n + 1.
    order = 2 * n + 1

    # Ensure edges are in a tuple for firls.
    edges = tuple(edges)

    # Use scipy's Parks–McClellan (firls) to compute linear‑phase FIR coefficients.
    # Frequency specification: 0.0 -> 1.0 with passband (edges) and stopband [0,1].
    # Desired values: 1 in passband, 0 in stopband.
    coeffs = signal.firls(order, (0.0, *edges, 1.0), [1, 1, 0, 0])

    return coeffs