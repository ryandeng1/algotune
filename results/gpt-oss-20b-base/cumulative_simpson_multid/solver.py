import numpy as np
from scipy.integrate import cumulative_simpson

def solve(problem: dict) -> np.ndarray:
    """Fast cumulative Simpson integral along the last axis."""
    y2 = problem["y2"]
    dx = problem["dx"]
    # avoid extra overhead by using local references
    return cumulative_simpson(y2, dx=dx)