import numpy as np
from sklearn.utils.extmath import randomized_svd

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Computes an approximate randomized SVD of the input matrix.

        Parameters
        ----------
        problem : dict
            Must contain the keys:
                - "matrix" : Iterable of shape (n, m)
                - "n_components" : int, number of singular values/vectors to compute
                - "matrix_type" : str, used to choose the number of power iterations
        kwargs
            Additional keyword arguments are ignored.

        Returns
        -------
        dict
            Keys:
                "U" : ndarray of shape (n, k)
                "S" : ndarray of shape (k,)
                "V" : ndarray of shape (m, k)
        """
        # Extract data
        A = np.asarray(problem["matrix"], dtype=np.float64)
        k = int(problem["n_components"])
        matrix_type = problem.get("matrix_type", "")

        # Determine number of power iterations
        if matrix_type == "ill_conditioned":
            n_iter = 10
        else:
            n_iter = 5

        # Compute randomized SVD
        U, s, Vt = randomized_svd(
            A, n_components=k, n_iter=n_iter, random_state=42
        )

        return {"U": U, "S": s, "V": Vt.T}
