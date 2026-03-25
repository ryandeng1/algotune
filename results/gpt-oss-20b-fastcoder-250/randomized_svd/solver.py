# solver.py
import numpy as np
from sklearn.utils.extmath import randomized_svd


class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Compute an approximate truncated singular value decomposition using a
        randomized algorithm. The returned matrices satisfy the orthogonality
        constraints and produce a reasonable approximation of the input matrix.
        """
        A = np.asarray(problem["matrix"], dtype=float)

        # Safety checks and defaults
        n_components = int(problem.get("n_components", 0))
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer")

        # Adjust the number of power iterations if the matrix is ill‑conditioned
        matrix_type = problem.get("matrix_type", "generic")
        n_iter = 10 if matrix_type == "ill_conditioned" else 5

        # Perform randomized SVD
        U, s, Vt = randomized_svd(
            A,
            n_components=n_components,
            n_iter=n_iter,
            random_state=42,
            # Note: `return_full_matrices=False` is the default and returns the
            # compact form directly.
        )

        # Return the result in the expected format
        return {"U": U, "S": s, "V": Vt.T}
