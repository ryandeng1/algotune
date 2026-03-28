import numpy as np
import ot


class Solver:
    """
    Fast solver for the Earth Mover's Distance (EMD) problem.
    Utilises the highly optimised `ot.emd` routine from
    the POT (Python Optimal Transport) library.
    """

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        """
        Compute the optimal transport plan between two discrete
        distributions using an efficient linear programming solver.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Must contain the keys:
                - 'source_weights': 1‑D array of shape (n,)
                - 'target_weights': 1‑D array of shape (m,)
                - 'cost_matrix'  : 2‑D array of shape (n, m)

        Returns
        -------
        dict[str, list[list[float]]]
            Dictionary with a single entry 'transport_plan' whose value
            is a nested Python list containing the optimal matrix G.
        """
        a = problem["source_weights"]
        b = problem["target_weights"]
        # Ensure the cost matrix is contiguous and float64 for POT
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)

        # Compute the optimal transport plan
        G = ot.emd(a, b, M, check_marginals=False)

        # Convert to a plain Python list of lists for the expected output
        return {"transport_plan": G.tolist()}