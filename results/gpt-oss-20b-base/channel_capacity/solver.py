import math
import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Computes the capacity of a discrete memoryless channel using the
        Blahut–Arimoto algorithm.

        Parameters
        ----------
        problem: dict
            Must contain ``"P"`` – an (m x n) probability transition matrix
            where ``P[i, j]`` = Pr(Y = i | X = j).
        kwargs: Any
            Unused keyword arguments.

        Returns
        -------
        dict
            Contains:
                * ``x`` – optimal input probability distribution (list of n floats)
                * ``C`` – channel capacity (float)
            Returns ``None`` if the input is invalid.
        """
        try:
            P = np.array(problem["P"], dtype=np.float64)
        except (KeyError, ValueError, TypeError):
            return None

        if P.ndim != 2:
            return None
        m, n = P.shape
        if m == 0 or n == 0:
            return None

        # Check that columns sum to 1 (within tolerance)
        if not np.allclose(P.sum(axis=0), 1.0, atol=1e-6):
            return None

        # Pad the log 0 = -inf handling
        def safe_log(p):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log(p, where=p > 0)

        ln2 = math.log(2.0)

        # Blahut–Arimoto initialization
        x = np.full(n, 1.0 / n, dtype=np.float64)
        # Precompute log P
        logP = safe_log(P)  # shape (m, n)

        # Iteration parameters
        max_iter = 10000
        tol = 1e-12

        for _ in range(max_iter):
            # Compute log Q_j = sum_i x_i * log P_{i,j}
            # Use broadcasting for efficiency
            logQ = logP @ x  # shape (m,)

            # Update y_i = sum_j P_{i,j} * x_j (output distribution)
            y = P @ x  # shape (m,)

            # Update x_j': proportional to exp( sum_i P_{i,j} * log(P_{i,j} / y_i) )
            # Compute numerator: exp( sum_i P_{i,j} * (logP_{i,j} - log y_i) )
            log_term = logP - safe_log(y[:, None])  # shape (m, n)
            exp_term = np.exp(log_term)  # shape (m, n)
            # Compute product over i: product(P_{i,j} / y_i)^(P_{i,j})
            # Equivalent to exp( sum_i P_{i,j} * log_term )
            prod = exp_term.sum(axis=0)  # sum_i P_{i,j} * log(P_{i,j}/y_i)
            new_x = prod
            new_x = new_x / new_x.sum()

            diff = np.max(np.abs(new_x - x))
            x = new_x
            if diff < tol:
                break

        # Compute capacity
        # Mutual information I = sum_{i,j} x_j * P_{i,j} * log2(P_{i,j} / y_i)
        # with y = P @ x
        y = P @ x
        # Avoid log(0) by masking
        mask = y > 0
        I = 0.0
        if mask.any():
            I_terms = P[mask] * x  # broadcasting works
            denom = y[mask][:, None]
            numerator_log = safe_log(P[mask])
            denom_log = safe_log(denom)
            I_terms = I_terms * (numerator_log - denom_log) / ln2
            I = I_terms.sum()

        return {"x": x.tolist(), "C": float(I)}
