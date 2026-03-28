import numpy as np
from scipy import signal

def solve(problem: tuple[int, tuple[float, float]]) -> np.ndarray:
    """
    Compute FIR filter coefficients with the `scipy.signal.firls` function.
    
    Parameters
    ----------
    problem : tuple[int, tuple[float, float]]
        A tuple containing the filter order part `n` and a pair of cutoff
        frequencies `(low, high)` to be used as the transition band edges.
    
    Returns
    -------
    np.ndarray
        The array of filter coefficients.
    """
    n, edges = problem                    # n is the filter order part
    # The length of the filter is `2*n + 1` (the default for firls)
    filt_len = 2 * n + 1
    # Directly build the breakpoints: 0, low, high, 1
    # and the desired amplitude at each segment
    breaks = (0.0, edges[0], edges[1], 1.0)
    amps  = (1, 1, 0, 0)
    # Compute coefficients
    coeffs = signal.firls(filt_len, breaks, amps)
    return coeffs