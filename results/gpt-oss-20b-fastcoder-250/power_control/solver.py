# solver.py
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

class Solver:
    """
    Fast solver for the optimal power control problem described in the task.
    Uses fixed‑point iteration of the standard interference function that
    gives the unique optimal solution for the linear fractional constraints.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the power control problem.

        Parameters
        ----------
        problem : dict
            Dictionary containing the problem data:
                "G"    : (n, n) array of path gains
                "σ"    : (n,) array of receiver noise powers
                "P_min": (n,) minimal transmitter powers
                "P_max": (n,) maximal transmitter powers
                "S_min": scalar minimal SINR (S_min > 0)

        Returns
        -------
        dict
            Dictionary with the keys:
                "P" : optimal transmitter powers (list of floats)
                "objective" : sum of optimal powers (float)
        """
        # Convert inputs to numpy arrays
        G: np.ndarray = np.asarray(problem["G"], dtype=np.float64)
        sigma: np.ndarray = np.asarray(problem["σ"], dtype=np.float64)
        P_min: np.ndarray = np.asarray(problem["P_min"], dtype=np.float64)
        P_max: np.ndarray = np.asarray(problem["P_max"], dtype=np.float64)
        S_min: float = float(problem["S_min"])

        n = G.shape[0]
        # Normalized channel gains: H_ik = G_ik / G_ii
        H = G / np.diag(G)[:, None]
        # Interference coefficient matrix excluding self
        A = H - np.eye(n)

        # Precompute constants: c = sigma / G_ii
        c = sigma / np.diag(G)

        # Initial guess: start at P_min
        P = P_min.copy()

        # Fixed point iteration parameters
        max_iter = 5000
        tol = 1e-12

        # Core iteration
        for _ in range(max_iter):
            # Interference term: A @ P
            inter = A @ P
            # Right‑hand side of the fixed‑point equation
            rhs = (c + inter) * S_min
            # Update power: new_P = RHS / (1 - S_min)
            # Actually from G_ii*P_i >= S_min*(sigma_i + sum_{k!=i}G_ik*P_k)
            # => P_i >= (sigma_i + sum_{k!=i}G_ik*P_k) * S_min / G_ii
            new_P = rhs * np.diag(G)  # bring back G_ii multiplication
            # Enforce bounds
            new_P = np.maximum(new_P, P_min)
            new_P = np.minimum(new_P, P_max)

            if np.linalg.norm(new_P - P, np.inf) < tol:
                P = new_P
                break
            P = new_P

        # Ensure we return a Python list
        return {"P": P.tolist(), "objective": float(P.sum())}
