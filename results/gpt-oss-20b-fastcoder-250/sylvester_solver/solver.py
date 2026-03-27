import numpy as np
from scipy.linalg import solve_sylvester

def solve(problem: dict) -> dict:
    """
    Solve the continuous-time Sylvester equation A X + X B = Q.

    Parameters
    ----------
    problem : dict
        Dictionary with keys "A", "B" and "Q". Each value is a NumPy array.
    Returns
    -------
    dict
        Dictionary containing the solution matrix under the key "X".
    """
    A = problem["A"]
    B = problem["B"]
    Q = problem["Q"]

    # Ensure the inputs are NumPy arrays for optimal performance
    A = np.asarray(A, dtype=np.result_type(A, Q, B))
    B = np.asarray(B, dtype=A.dtype)
    Q = np.asarray(Q, dtype=A.dtype)

    X = solve_sylvester(A, B, Q)
    return {"X": X}