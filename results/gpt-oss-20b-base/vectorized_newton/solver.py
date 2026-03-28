import numpy as np

# Coefficients used in the root finding problem
A2 = 1e-9
A3 = 0.004
A4 = 10.0
A5 = 0.27456


def _f(x, a0, a1):
    """Vectorised function values for all roots."""
    return (
        (x - a0)
        * np.exp(A2 * x)
        * (A3 * np.exp(-A4 * x) + A5)
        - a1
    )


def _fprime(x, a0, a1):
    """Vectorised derivative of the function."""
    e2 = np.exp(A2 * x)
    e4 = np.exp(-A4 * x)

    term1 = e2 * (A3 * e4 + A5)

    # Derivative of (x - a0)
    deriv1 = term1

    # Derivative of exp(A2*x)
    deriv2 = (x - a0) * A2 * e2 * (A3 * e4 + A5)

    # Derivative of (A3*exp(-A4*x)+A5)
    deriv3 = (x - a0) * e2 * (-A3 * A4 * e4)

    return deriv1 + deriv2 + deriv3


class Solver:
    def __init__(self):
        pass  # No state needed

    def solve(self, problem: dict[str, list[float]]) -> dict[str, list[float]]:
        """
        Find roots for each triplet (x0, a0, a1) using a fast vectorised Newton
        method that does not rely on scipy.  Returns a list of roots.
        On failure the corresponding entry is NaN.
        """
        try:
            x0_arr = np.asarray(problem["x0"], dtype=float)
            a0_arr = np.asarray(problem["a0"], dtype=float)
            a1_arr = np.asarray(problem["a1"], dtype=float)

            if not (len(x0_arr) == len(a0_arr) == len(a1_arr)):
                raise ValueError("Input lists must have the same length")
        except Exception:  # pragma: no cover
            return {"roots": []}

        n = x0_arr.size

        # Initial guess
        x = x0_arr.copy()

        # Parameters for the Newton iterations
        it_max = 50
        eps = 1e-14

        # Perform Newton iterations in a vectorised fashion
        for _ in range(it_max):
            fx = _f(x, a0_arr, a1_arr)
            if np.all(np.abs(fx) < eps):
                break
            fpx = _fprime(x, a0_arr, a1_arr)

            # Avoid division by zero; if derivative is zero set a tiny value
            fpx_safe = np.where(fpx == 0, 1e-18, fpx)

            x -= fx / fpx_safe

        # Any remaining not converged values are set to NaN
        not_converged = np.abs(_f(x, a0_arr, a1_arr)) >= eps
        x[not_converged] = np.nan

        # Ensure the output is a plain Python list of floats
        roots = x.tolist()
        if len(roots) != n:
            roots.extend([float("nan")] * (n - len(roots)))  # safety net

        return {"roots": roots}