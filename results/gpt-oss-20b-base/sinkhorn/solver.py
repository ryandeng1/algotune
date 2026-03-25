# solver.py
import numpy as np
from typing import Any, Dict

class Solver:
    """Fast entropic OT solver using log‑stable Sinkhorn iterations."""

    def _sinkhorn(
        self,
        a: np.ndarray,
        b: np.ndarray,
        M: np.ndarray,
        reg: float,
        max_iter: int = 500,
        eps: float = 1e-9
    ) -> np.ndarray:
        """
        Computes the entropically regularized optimal transport plan
        using the stable Sinkhorn algorithm.
        """
        # Pre‑compute the kernel matrix in log‑domain
        K = np.exp(-M / reg)  # shape (n, m)
        u = np.ones_like(a)
        v = np.ones_like(b)

        for _ in range(max_iter):
            u_prev = u.copy()
            # Update u and v
            u = a / (K @ v)
            v = b / (K.T @ u)

            # Check convergence on marginals
            if np.all(np.abs(u - u_prev) < eps):
                break

        # Final transport plan
        G = (u[:, None] * K) * v[None, :]
        return G

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Computes the entropic optimal transport plan for the given problem.
        Returns a dictionary with key 'transport_plan'.
        """
        try:
            a = np.asarray(problem["source_weights"], dtype=np.float64)
            b = np.asarray(problem["target_weights"], dtype=np.float64)
            M = np.asarray(problem["cost_matrix"], dtype=np.float64, order='C')
            reg = float(problem["reg"])

            G = self._sinkhorn(a, b, M, reg, max_iter=1000, eps=1e-9)

            if not np.isfinite(G).all():
                raise ValueError("Non‑finite values in transport plan")

            return {"transport_plan": G, "error_message": None}
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}
