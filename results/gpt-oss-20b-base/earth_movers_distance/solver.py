import numpy as np
import ot


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        """
        Solve the Earth Mover's Distance (EMD) problem using POT's LP solver.
        The function expects the input problem dictionary to contain
        the source weights `source_weights` (1‑D array-like), the target weights
        `target_weights` (1‑D array-like), and the cost matrix `cost_matrix`
        (2‑D array-like). The source and target weight vectors must sum to the
        same value and be non‑negative. The returned solution dictionary
        contains the optimal transport plan under the key `"transport_plan"`.
        """
        a = np.asarray(problem["source_weights"], dtype=np.float64, order="C").ravel()
        b = np.asarray(problem["target_weights"], dtype=np.float64, order="C").ravel()
        M = np.asarray(problem["cost_matrix"], dtype=np.float64, order="C")

        # The POT `emd` implementation requires a C‑contiguous matrix.
        G = ot.lp.emd(a, b, M, check_marginals=False)

        # Convert to a Python-friendly list of lists
        return {"transport_plan": G.tolist()}