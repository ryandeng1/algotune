import numpy as np
from scipy.linalg import qz

def solve(problem: dict[str, list[list[float]]]) -> dict[str, dict[str, list[list[float | complex]]]]:
    """
    Compute the real QZ factorization of (A, B):
        A = Q @ AA @ Z.T
        B = Q @ BB @ Z.T
    The inputs are lists of lists; the output is nested lists as required by the
    specification.
    """
    # Convert inputs to NumPy arrays in a single call for speed
    A = np.array(problem["A"], dtype=float, copy=False)
    B = np.array(problem["B"], dtype=float, copy=False)

    # Perform the QZ factorization
    AA, BB, Q, Z = qz(A, B, output="real")

    # Convert the results back to plain Python lists
    return {
        "QZ": {
            "AA": AA.tolist(),
            "BB": BB.tolist(),
            "Q": Q.tolist(),
            "Z": Z.tolist(),
        }
    }