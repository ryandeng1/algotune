# solver.py
from typing import Any, Dict, List, Union
import numpy as np
import ot


class Solver:
    """
    Solver for optimal transport problems using entropic regularisation.

    The implementation relies on the `pytorch-ot` library which provides a
    highly‑optimised C implementation of the Sinkhorn algorithm.
    """

    def solve(
        self, problem: Dict[str, Any]
    ) -> Union[
        Dict[str, Union[List[List[float]], None, str]],
        dict,
    ]:
        """
        Compute the entropic regularised optimal transport plan.

        Parameters
        ----------
        problem: dict
            Dictionary containing:
                - source_weights: list[float]
                - target_weights: list[float]
                - cost_matrix: 2‑D list[float] or np.ndarray
                - reg: float

        Returns
        -------
        dict
            {
                'transport_plan': 2‑D numpy array (or None on error),
                'error_message': None (if successful) or error string
            }
        """
        # --- Input conversion -------------------------------------------------
        a = np.asarray(problem["source_weights"], dtype=np.float64, copy=False)
        b = np.asarray(problem["target_weights"], dtype=np.float64, copy=False)
        M = np.asarray(problem["cost_matrix"], dtype=np.float64, order="C")
        reg = float(problem["reg"])

        # --- Solve -------------------------------------------------------------
        try:
            # ot.sinkhorn is fast C++ code with optional Numba acceleration.
            # The function expects a, b as 1‑D arrays and the cost matrix as
            # a 2‑D contiguous array.
            G = ot.sinkhorn(a, b, M, reg, numItermax=1000, stopThr=1e-9)
            if not np.isfinite(G).all():
                raise ValueError("Non‑finite values in transport plan")
            return {"transport_plan": G, "error_message": None}
        except Exception as exc:  # pragma: no cover
            return {"transport_plan": None, "error_message": str(exc)}