import numpy as np
from numbers import Real
from typing import Any
from sklearn.neighbors import KernelDensity
from sklearn.exceptions import NotFittedError

class Solver:
    available_kernels = {"gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"}

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        try:
            # fast conversion to numpy arrays, no copy if already array
            X = np.asarray(problem["data_points"], dtype=float)
            Q = np.asarray(problem["query_points"], dtype=float)

            # validate shapes
            if X.ndim != 2 or Q.ndim != 2:
                raise ValueError("Both data points and query points must be 2‑D arrays.")

            n, d = X.shape
            if n == 0:
                raise ValueError("No data points provided.")
            if Q.shape[0] == 0:
                return {"log_density": []}
            if d != Q.shape[1]:
                raise ValueError("Dimension mismatch between data and query points.")

            # validate bandwidth
            bw = problem["bandwidth"]
            if not isinstance(bw, Real) or bw <= 0:
                raise ValueError("Bandwidth must be a positive real number.")
            bw = float(bw)

            kernel = problem["kernel"]
            if kernel not in self.available_kernels:
                raise ValueError(f"Unknown kernel: {kernel}")

            # fit KDE and evaluate
            kde = KernelDensity(kernel=kernel, bandwidth=bw)
            kde.fit(X)
            scores = kde.score_samples(Q)

            return {"log_density": scores.tolist()}

        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except (ValueError, TypeError, NotFittedError, np.linalg.LinAlgError) as e:
            return {"error": f"Computation error: {e}"}
        except Exception as e:  # pragma: no cover
            return {"error": f"Unexpected error: {e}"}