# solver.py
# -------------- 1. Imports ---------------
import numpy as np
import cvxpy as cp
from typing import Dict, Any


# -------------- 2. Solver Class ---------------
class Solver:
    """
    Tensor completion via low‑rank nuclear norm minimisation.

    The implementation below replaces the original triple‑nested loops
    with efficient NumPy operations, eliminates unreachable code, and
    uses the fastest available solver for CVXPY (SCS with the ``high
    quality`` option).  Execution time will be dominated by the convex
    optimisation, not by data preparation.
    """

    # ------------------------------------------------------------------
    def _unfold(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """
        Unfold the tensor along the specified mode.

        Parameters
        ----------
        tensor : np.ndarray
            The input tensor.
        mode : int
            0 → mode‑1, 1 → mode‑2, 2 → mode‑3.

        Returns
        -------
        np.ndarray
            Unfolded matrix of shape (dim_mode, prod(other_dims)).
        """
        return np.moveaxis(tensor, mode, 0).reshape(
            tensor.shape[mode], -1
        )

    # ------------------------------------------------------------------
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the tensor completion problem.

        Parameters
        ----------
        problem : dict
            Dictionary with keys ``tensor`` and ``mask``.  Both contain
            nested lists that describe a 3‑D array where some entries are
            missing.

        Returns
        -------
        dict
            Dictionary with key ``completed_tensor``.  The value is a
            list representation of the completed tensor.
        """
        # --------------------------------------------------------------
        # 1. Convert inputs to NumPy arrays and determine shape
        # --------------------------------------------------------------
        observed = np.asarray(problem["tensor"], dtype=np.float64)
        mask = np.asarray(problem["mask"], dtype=bool)

        if observed.shape != mask.shape:
            return {"completed_tensor": []}

        # --------------------------------------------------------------
        # 2. Pre‑compute unfoldings and mask matrices
        # --------------------------------------------------------------
        U1, U2, U3 = (
            self._unfold(observed, 0),
            self._unfold(observed, 1),
            self._unfold(observed, 2),
        )
        M1, M2, M3 = (
            self._unfold(mask, 0),
            self._unfold(mask, 1),
            self._unfold(mask, 2),
        )
        # --------------------------------------------------------------
        # 3. CVXPY variables and optimisation problem
        # --------------------------------------------------------------
        X1 = cp.Variable(U1.shape)
        X2 = cp.Variable(U2.shape)
        X3 = cp.Variable(U3.shape)

        objective = cp.Minimize(
            cp.normNuc(X1) + cp.normNuc(X2) + cp.normNuc(X3)
        )

        constraints = [
            cp.multiply(X1, M1) == cp.multiply(U1, M1),
            cp.multiply(X2, M2) == cp.multiply(U2, M2),
            cp.multiply(X3, M3) == cp.multiply(U3, M3),
        ]

        prob = cp.Problem(objective, constraints)

        # --------------------------------------------------------------
        # 4. Solve using the fastest CVXPY solver available
        # --------------------------------------------------------------
        try:
            # SCS is the default solver for nuclear‑norm problems.
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=10000)
        except (cp.SolverError, Exception):
            return {"completed_tensor": []}

        # --------------------------------------------------------------
        # 5. Extract result and reshape to original dimensions
        # --------------------------------------------------------------
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or X1.value is None:
            return {"completed_tensor": []}

        result = X1.value.reshape(observed.shape)
        return {"completed_tensor": result.tolist()}