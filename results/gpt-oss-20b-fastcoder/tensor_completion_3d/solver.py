#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any
import numpy as np
import cvxpy as cp


class Solver:
    """Optimised tensor‑completion solver."""

    @staticmethod
    def _unfold(tensor: np.ndarray, axis: int) -> np.ndarray:
        """
        Fast tensor unfolding along a chosen axis using numpy reshape and transpose.
        """
        # Move the chosen axis to the front
        return np.transpose(tensor, (axis, *range(0, axis), *range(axis + 1, tensor.ndim))).reshape(
            tensor.shape[axis], -1
        )

    @staticmethod
    def _unfold_mask(mask: np.ndarray, axis: int) -> np.ndarray:
        """Unfold the boolean mask along the same axis as the tensor."""
        return Solver._unfold(mask, axis)

    def solve(self, problem: dict) -> dict:
        """
        Solve the tensor completion problem using nuclear‑norm minimisation.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - 'tensor': the partially observed tensor (list of lists…).
                - 'mask'  : boolean mask of observed entries (same shape).

        Returns
        -------
        dict
            Completed tensor as a nested list under key 'completed_tensor'.
        """
        # Convert inputs to efficient numpy arrays
        observed = np.array(problem["tensor"], dtype=float)
        mask = np.array(problem["mask"], dtype=bool)

        # Ensure no premature allocation of large zero arrays.
        dim1, dim2, dim3 = observed.shape

        # Unfold the tensor and the mask along three modes
        unfolding1, mask1 = self._unfold(observed, 0), self._unfold_mask(mask, 0)
        unfolding2, mask2 = self._unfold(observed, 1), self._unfold_mask(mask, 1)
        unfolding3, mask3 = self._unfold(observed, 2), self._unfold_mask(mask, 2)

        # Variables for the three unfolding modes
        X1 = cp.Variable(unfolding1.shape)
        X2 = cp.Variable(unfolding2.shape)
        X3 = cp.Variable(unfolding3.shape)

        # Objective: sum of trace (nuclear) norms
        objective = cp.Minimize(
            cp.norm(X1, "nuc") + cp.norm(X2, "nuc") + cp.norm(X3, "nuc")
        )

        # Constraints: match known entries in each unfolding
        constraints = [
            cp.multiply(X1, mask1) == cp.multiply(unfolding1, mask1),
            cp.multiply(X2, mask2) == cp.multiply(unfolding2, mask2),
            cp.multiply(X3, mask3) == cp.multiply(unfolding3, mask3),
        ]

        # Build and solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)  # use a fast solver
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or X1.value is None:
                return {"completed_tensor": []}
            # Reconstruct the tensor from one unfolding
            completed = X1.value.reshape(observed.shape)
            return {"completed_tensor": completed.tolist()}
        except (cp.SolverError, Exception):
            return {"completed_tensor": []}