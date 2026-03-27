import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KernelDensity
from numbers import Real
from typing import Any

class Solver:
    # cache kernels for quick membership test
    _available_kernels = {"gaussian", "tophat", "epanechnikov", "exponential",
                          "linear", "cosine"}

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        try:
            # Convert to numpy arrays (no copy if already array)
            X = np.asarray(problem["data_points"], dtype=float)
            Xq = np.asarray(problem["query_points"], dtype=float)

            if X.ndim != 2 or Xq.ndim != 2:
                raise ValueError("Data points or query points must be 2‑D.")
            if X.size == 0:
                raise ValueError("No data points provided.")
            if Xq.size == 0:
                return {"log_density": []}
            if X.shape[1] != Xq.shape[1]:
                raise ValueError("Mismatched dimensionality.")

            kernel = problem["kernel"]
            bandwidth = problem["bandwidth"]

            if not isinstance(bandwidth, Real) or bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            if kernel not in self._available_kernels:
                raise ValueError(f"Unknown kernel: {kernel}")

            kde = KernelDensity(kernel=kernel, bandwidth=float(bandwidth))
            kde.fit(X)
            log_density = kde.score_samples(Xq)
            return {"log_density": log_density.tolist()}

        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except (ValueError, TypeError, NotFittedError, np.linalg.LinAlgError) as e:
            return {"error": f"Computation error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}