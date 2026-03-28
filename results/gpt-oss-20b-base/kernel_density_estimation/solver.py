import numpy as np
from sklearn.neighbors import KernelDensity
from numbers import Real
from typing import Any

class Solver:
    def __init__(self):
        self._kernels = {
            'gaussian': 'gaussian',
            'tophat': 'tophat',
            'epanechnikov': 'epanechnikov',
            'exponential': 'exponential',
            'linear': 'linear',
            'cosine': 'cosine',
        }

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Compute log‑density estimates at query points using sklearn's KernelDensity.

        The function strictly validates input and returns results in a dict.  All
        exceptions are caught and translated into an error message.
        """
        try:
            # Data extraction
            X = np.asarray(problem['data_points'], dtype=float)
            X_q = np.asarray(problem['query_points'], dtype=float)
            kernel = problem['kernel']
            bw = problem['bandwidth']

            # Basic shape checks
            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError('Data points or query points must be 2‑D arrays.')
            if X.shape[0] == 0:
                raise ValueError('The data set is empty.')
            if X_q.shape[0] == 0:
                return {'log_density': []}
            if X.shape[1] != X_q.shape[1]:
                raise ValueError('Input dimensions mismatch between data and queries.')

            # Check bandwidth
            if not isinstance(bw, Real) or bw <= 0:
                raise ValueError('Bandwidth must be a positive real number.')
            bw = float(bw)

            # Kernel validation
            if kernel not in self._kernels:
                raise ValueError('Unknown kernel requested.')

            # Fit density estimator
            kde = KernelDensity(kernel=kernel, bandwidth=bw)
            kde.fit(X)

            # Evaluate at query points
            logp = kde.score_samples(X_q)
            return {'log_density': logp.tolist()}

        except KeyError as k:
            return {'error': f'Missing key: {k}'}
        except Exception as e:
            return {'error': f'Computation error: {e}'}