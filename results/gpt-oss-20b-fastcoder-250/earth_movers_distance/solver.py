import numpy as np
import ot


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        """Solve the EMD problem with POT in an efficient way."""

        # Pull data from the problem dictionary; they are already NumPy arrays
        a = problem["source_weights"]
        b = problem["target_weights"]
        M = problem["cost_matrix"]

        # POT requires float64 C‑contiguous arrays – ensure this in one step
        M = np.require(M, dtype=np.float64, requirements=["C_CONTIGUOUS"])

        # Compute the optimal transport plan
        # `check_marginals=False` speeds up the call and is safe because the
        # generated problems always satisfy equal mass.
        G = ot.emd(a, b, M, check_marginals=False)

        # Convert the result matrix to a python list of lists for the API
        return {"transport_plan": G.tolist()}