import numpy as np

class Solver:
    """
    Vectorised Newton‑Raphson solver for the scalar equation

        f(x, a0,...,a5) =
            a1 - a2*(exp((a0 + x*a3)/a5) - 1)
            - (a0 + x*a3)/a4 - x
    """

    def __init__(self,
                 a2: float = 1.0,
                 a3: float = 2.0,
                 a4: float = 3.0,
                 a5: float = 4.0):
        self.a2 = float(a2)
        self.a3 = float(a3)
        self.a4 = float(a4)
        self.a5 = float(a5)

    # ------------------------------------------------------------------
    def _f(self, x: np.ndarray, a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
        """Evaluate f at x for each instance."""
        z = (a0 + x * self.a3) / self.a5
        exp_z = np.exp(z)
        return a1 - self.a2 * (exp_z - 1.0) - (a0 + x * self.a3) / self.a4 - x

    # ------------------------------------------------------------------
    def _df(self, x: np.ndarray, a0: np.ndarray) -> np.ndarray:
        """Evaluate derivative f' at x for each instance."""
        z = (a0 + x * self.a3) / self.a5
        exp_z = np.exp(z)
        # derivative of exp part: exp_z * a3 / a5
        return -self.a2 * exp_z * self.a3 / self.a5 - self.a3 / self.a4 - 1.0

    # ------------------------------------------------------------------
    def solve(self, problem: dict[str, list[float]]) -> dict[str, np.ndarray]:
        """
        Solve for roots of the given equation for many instances in
        a vectorised manner.

        Parameters
        ----------
        problem : dict
            Must contain keys "x0", "a0" and "a1". Each maps to a list
            of length n. We convert them to contiguous float64 arrays.

        Returns
        -------
        dict
            {"roots": ndarray} where the ndarray has shape (n,)
            containing the approximate root for each instance.
            If a single root could not be found, NaN is returned for
            that instance.
        """
        # ------------------------------------------------------------------
        # Convert inputs to numpy arrays
        try:
            x = np.asarray(problem["x0"], dtype=np.float64)
            a0 = np.asarray(problem["a0"], dtype=np.float64)
            a1 = np.asarray(problem["a1"], dtype=np.float64)
        except Exception:
            return {"roots": np.array([], dtype=np.float64)}

        if not (x.shape == a0.shape == a1.shape):
            return {"roots": np.full_like(x, np.nan, dtype=np.float64)}

        # ------------------------------------------------------------------
        # Newton‑Raphson parameters
        tol = 1e-10          # tolerance for convergence
        max_iter = 80        # safety limit
        n = x.size

        # ------------------------------------------------------------------
        # Initial guess
        roots = x.copy()
        # Keep mask of converged entries
        not_converged = np.arange(n, dtype=bool)

        for _ in range(max_iter):
            # Evaluate function and derivative only for not converged
            if not np.any(not_converged):
                break

            fx = self._f(roots[not_converged], a0[not_converged], a1[not_converged])
            dfx = self._df(roots[not_converged], a0[not_converged])

            # Prevent division by zero
            zero_d = dfx == 0
            if np.any(zero_d):
                # Assign NaN to those that hit zero derivative
                not_converged[not_converged][zero_d] = False
                roots[not_converged][zero_d] = np.nan
                if not np.any(not_converged):
                    break

            delta = fx / dfx
            roots[not_converged] -= delta

            # Check convergence
            new_not_converged = np.abs(delta) > tol
            # Update mask
            indices = np.where(not_converged)[0]
            not_converged[indices[new_not_converged]] = True
            not_converged[indices[~new_not_converged]] = False

        # Any remaining non‑converged entries get NaN
        roots[not_converged] = np.nan

        return {"roots": roots}
