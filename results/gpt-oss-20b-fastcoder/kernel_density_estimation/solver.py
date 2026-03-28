import numpy as np
from numbers import Real
from sklearn.neighbors import KernelDensity
from sklearn.exceptions import NotFittedError

class Solver:
    available_kernels = {"gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"}

    def solve(self, problem: dict) -> dict:
        try:
            X = np.asarray(problem["data_points"], dtype=float)
            Xq = np.asarray(problem["query_points"], dtype=float)
            if X.ndim != 2 or Xq.ndim != 2:
                raise ValueError("Data points or query points must be 2‑D arrays.")
            if X.size == 0:
                raise ValueError("No data points provided.")
            if Xq.size == 0:
                return {"log_density": []}
            if X.shape[1] != Xq.shape[1]:
                raise ValueError("Data points and query points have different dimensions.")
            bw = problem["bandwidth"]
            if not isinstance(bw, Real) or bw <= 0:
                raise ValueError("Bandwidth must be positive.")
            kernel = problem["kernel"]
            if kernel not in self.available_kernels:
                raise ValueError(f"Unknown kernel: {kernel}")

            kde = KernelDensity(kernel=kernel, bandwidth=float(bw))
            kde.fit(X)
            log_dens = kde.score_samples(Xq)
            return {"log_density": log_dens.tolist()}
        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except (ValueError, TypeError, NotFittedError, np.linalg.LinAlgError) as e:
            return {"error": f"Computation error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}