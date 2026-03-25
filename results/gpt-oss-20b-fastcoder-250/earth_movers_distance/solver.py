# solver.py
import numpy as np
import ot

class Solver:
    def solve(self, problem: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        """
        Compute the optimal transport plan for the Earth Mover's Distance problem
        using the POT (Python Optimal Transport) library.

        The solution is returned in the same structured format as the reference
        implementation: a dictionary with a single key 'transport_plan'
        containing the coupling matrix as a NumPy array.

        Parameters
        ----------
        problem : dict
            Dictionary with keys 'source_weights', 'target_weights', and
            'cost_matrix'.

        Returns
        -------
        dict
            {'transport_plan': G}
        """
        # Extract inputs
        a = problem["source_weights"]
        b = problem["target_weights"]
        M = problem["cost_matrix"]

        # POT's emd expects a C-contiguous float64 matrix
        M_cont = np.ascontiguousarray(M, dtype=np.float64)

        # Compute the optimal coupling. The check_marginals flag is False
        # because the problem generator guarantees that the total mass of
        # source and target distributions is equal.
        G = ot.lp.emd(a, b, M_cont, check_marginals=False)

        return {"transport_plan": G}
