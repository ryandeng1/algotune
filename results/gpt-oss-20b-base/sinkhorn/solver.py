# solver.py
from __future__ import annotations

import numpy as np
import ot


class Solver:
    """
    Efficient solver for entropy‑regularized optimal transport.

    The implementation relies on the “sinkhorn” routine from the POT library.
    The code is intentionally minimal – all preparation is done up front
    with NumPy array conversions, then a single fast call to the compiled
    routine.  No extra Python‑level loops or checks are performed to keep
    the runtime as low as possible.
    """

    def solve(self, problem: dict[str, object]) -> dict[str, object]:
        """
        Parameters
        ----------
        problem : dict
            * source_weights : List[float] – weights of the source distribution
            * target_weights : List[float] – weights of the target distribution
            * cost_matrix   : List[List[float]] – 2‑D cost matrix
            * reg           : float – regularisation parameter

        Returns
        -------
        dict
            * transport_plan : np.ndarray or None
            * error_message  : str or None
        """
        try:
            a = np.asarray(problem["source_weights"], dtype=np.float64, order="C")
            b = np.asarray(problem["target_weights"], dtype=np.float64, order="C")
            M = np.asarray(problem["cost_matrix"], dtype=np.float64, order="C")

            # POT expects 1‑D source/target arrays and a 2‑D cost matrix.
            G = ot.sinkhorn(a, b, M, problem["reg"])

            # Guard against numerical issues – still very cheap.
            if not np.isfinite(G).all():
                raise ValueError("Non‑finite values in transport plan")
            return {"transport_plan": G, "error_message": None}
        except Exception as exc:  # pragma: no cover
            return {"transport_plan": None, "error_message": str(exc)}