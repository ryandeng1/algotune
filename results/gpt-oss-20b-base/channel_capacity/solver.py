import numpy as np
import cvxpy as cp
from scipy.special import xlogy
import math

class Solver:
    def solve(self, problem: dict) -> dict:
        """Optimise the mutual information for a discrete memoryless channel.

        Parameters
        ----------
        problem : dict
            Must contain a key ``'P'`` mapping to a 2‑D array with shape
            ``(n, m)`` representing the channel transition matrix.

        Returns
        -------
        dict or None
            Returns ``{'x': list, 'C': float}`` where ``x`` is the capacity‑achieving
            input distribution and ``C`` is the channel capacity.  ``None`` is
            returned when the problem is malformed or the optimisation fails.
        """

        # ------------------------------------------------------------------
        # 1. Validate the input matrix.
        # ------------------------------------------------------------------
        try:
            P = np.asarray(problem["P"], dtype=np.float64)
        except (KeyError, ValueError):
            return None

        if P.ndim != 2:
            return None

        n, m = P.shape
        if n <= 0 or m <= 0 or P.shape != (n, m):
            return None

        # ------------------------------------------------------------------
        # 2. Set up the convex optimisation problem in CVXPY.
        # ------------------------------------------------------------------
        # Input probability vector.
        x = cp.Variable(n, name="x")

        # Channel output: y_j = sum_i P[j,i] * x_i
        y = P @ x

        # Pre‑compute the vector `c` that multiplies `x` in the Mutual Information.
        #   c_k = sum_j P[j,k] * log(P[j,k] / P[j,:] @ x)   (in bits)
        #
        # We use the identity
        #   log(P[j,k] / (P @ x)[j]) = log(P[j,k]) - log((P @ x)[j])
        # and replace the remaining term with `entr(P @ x)` later.
        # The expression is linear in `x` once the logarithm of the denominator
        # is taken into account via the `entr` function (which is concave).
        #
        # The coefficient vector `c` is constant and does not depend on `x`,
        # but it still needs to be evaluated symbolically in CVXPY for the
        # optimiser to recognise the structure.
        #
        # We approximate it numerically by evaluating `xlogy` with a small
        # perturbation on the current `x`.  This keeps the formulation
        # exactly the same (because `entr` is used instead of the logarithm
        # of the denominator).
        #
        # The `entr` function is defined as `-p*log(p)` and is concave for
        # `p >= 0`.  Using it keeps the objective convex (concave in the
        # minimisation sense, which CVXPY turns into a maximisation).
        c = np.sum(xlogy(P, P), axis=0) / math.log(2.0)

        # Mutual information in bits
        mutual_information = c @ x + cp.sum(cp.entr(y) / math.log(2.0))

        # Objective: maximise `I`
        objective = cp.Maximize(mutual_information)

        # Constraints: `x` is a probability vector
        constraints = [cp.sum(x) == 1.0, x >= 0]

        # ------------------------------------------------------------------
        # 3. Solve using a fast, but reasonably accurate, solver.
        # ------------------------------------------------------------------
        prob = cp.Problem(objective, constraints)
        try:
            # Use the SCS solver – it is typically faster for small problems
            # and is the default in CVXPY when the problem is unconstrained
            # by a specific solver.  We set a generous tolerance but ask the
            # solver to use only a single thread to avoid oversubscription.
            prob.solve(solver=cp.SCS, verbose=False, max_iters=1000, eps=1e-7, use_indirect=True)
        except (cp.SolverError, Exception):
            return None

        if prob.value is None or prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return None

        return {"x": x.value.tolist(), "C": prob.value}