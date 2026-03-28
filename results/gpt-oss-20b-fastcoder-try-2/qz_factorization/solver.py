import numpy as np
from scipy.linalg import qz

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> dict[str, dict[str, list[list[float | complex]]]]:
        # Convert input lists to contiguous numpy arrays
        A = np.asarray(problem['A'], dtype=np.float64, order='C')
        B = np.asarray(problem['B'], dtype=np.float64, order='C')
        # Perform QZ factorization (real Schur form)
        AA, BB, Q, Z = qz(A, B, output='real')
        # Convert back to nested Python lists
        return {
            'QZ': {
                'AA': AA.tolist(),
                'BB': BB.tolist(),
                'Q' : Q.tolist(),
                'Z' : Z.tolist()
            }
        }