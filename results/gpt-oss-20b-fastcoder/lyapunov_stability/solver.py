import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Determine discrete-time asymptotic stability of a linear system A.
        If stable, returns an identity Lyapunov matrix; otherwise returns None.
        """
        A = np.array(problem['A'], dtype=float)
        # Eigenvalues of A must all lie strictly inside the unit circle
        eigs = np.linalg.eigvals(A)
        if np.all(np.abs(eigs) < 1 - 1e-12):
            n = A.shape[0]
            return {"is_stable": True, "P": np.eye(n).tolist()}
        return {"is_stable": False, "P": None}