import numpy as np
import ot

class Solver:

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        """
        Solve the EMD problem using ot.lp.emd.

        :param problem: A dictionary representing the EMD problem.
        :return: A dictionary with key "transport_plan" containing the optimal
                 transport plan matrix G as a list of lists.
        """
        a = problem["source_weights"]
        b = problem["target_weights"]
        M = problem["cost_matrix"]
        # Ensure the cost matrix is contiguous in memory and of type float64
        M_cont = np.asarray(M, dtype=np.float64, order="C")
        G = ot.lp.emd(a, b, M_cont, check_marginals=False)
        return {"transport_plan": G}