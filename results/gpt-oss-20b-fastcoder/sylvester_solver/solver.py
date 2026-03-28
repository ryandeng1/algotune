import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = problem['A']
        B = problem['B']
        Q = problem['Q']

        # Vectorize the Sylvester equation AX + XB = Q
        # (I ⊗ A + Bᵀ ⊗ I) vec(X) = vec(Q)
        # Use the dense Kronecker form for speed
        n = A.shape[0]
        m = B.shape[0]
        # Construct the coefficient matrix explicitly (size nm x nm)
        IA = np.eye(m, dtype=A.dtype)
        I_ = np.eye(n, dtype=A.dtype)
        coef = np.kron(IA, A) + np.kron(B.T, I_)
        rhs = Q.reshape(-1, order='F')  # column-major vectorization

        # Solve the linear system
        vecX = np.linalg.solve(coef, rhs)

        # Reshape back to matrix form
        X = vecX.reshape((n, m), order='F')
        return {'X': X}