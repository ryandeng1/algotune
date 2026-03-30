# solver.py
import numpy as np
import ot

class Solver:
    """
    Efficient Sinkhorn‑like Solver for the Earth Mover's Distance (EMD)
    problem using the `ot.lp.emd` routine from the POT library.

    The implementation performs minimal copying and type conversions,
    and turns the resulting NumPy array plan into the list‑of‑lists form
    requested by the specifications.
    """

    @staticmethod
    def _to_float64_contiguous(x: np.ndarray) -> np.ndarray:
        """Return a contiguous float64 view of the input array."""
        return np.ascontiguousarray(x, dtype=np.float64)

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        """
        Compute the optimal transport plan G for the given EMD instance.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Must contain:
            - 'source_weights'  : numpy array a of shape (n,)
            - 'target_weights'  : numpy array b of shape (m,)
            - 'cost_matrix'     : numpy array M of shape (n, m)

        Returns
        -------
        dict[str, list[list[float]]]
            A dictionary with key 'transport_plan' whose value is the optimal
            transport plan G expressed as a list of lists of floats.
        """
        a = self._to_float64_contiguous(problem['source_weights'])
        b = self._to_float64_contiguous(problem['target_weights'])
        M = self._to_float64_contiguous(problem['cost_matrix'])

        # ot.lp.emd optionally verifies marginals; skipping this speeds up
        G = ot.lp.emd(a, b, M, check_marginals=False)

        # Convert the dense NumPy plan to the required list‑of‑lists format
        return {"transport_plan": G.tolist()}