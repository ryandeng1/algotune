import numpy as np
from sklearn.neighbors import KernelDensity
from numbers import Real

class Solver:
    # kernels supported by sklearn's KernelDensity
    available_kernels = {"gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"}

    def solve(self, problem: dict) -> dict:
        """Estimate log density at query points using KDE."""
        try:
            X = np.asarray(problem["data_points"], dtype=float)
            Xq = np.asarray(problem["query_points"], dtype=float)
            kernel = problem["kernel"]
            h = problem["bandwidth"]

            if X.ndim != 2 or Xq.ndim != 2:
                return {"error": "Data points or query points must be 2D arrays."}
            if X.shape[0] == 0:
                return {"error": "No data points provided."}
            if Xq.shape[0] == 0:
                return {"log_density": []}
            if X.shape[1] != Xq.shape[1]:
                return {"error": "Mismatched dimensionality between data and query points."}

            if not isinstance(h, Real) or h <= 0:
                return {"error": "Band width must be a positive real number."}
            if kernel not in self.available_kernels:
                return {"error": f"Unsupported kernel '{kernel}'."}

            kde = KernelDensity(kernel=kernel, bandwidth=float(h))
            kde.fit(X)
            log_dens = kde.score_samples(Xq)
            return {"log_density": log_dens.tolist()}

        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except Exception as e:
            return {"error": f"Computation error: {e}"}
