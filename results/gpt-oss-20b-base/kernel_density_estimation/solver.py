# solver.py

import numpy as np
from typing import Any

from sklearn.neighbors import KernelDensity
from sklearn.exceptions import NotFittedError
from numbers import Real

class Solver:
    def __init__(self) -> None:
        # kernels supported by scikit‑learn
        self._kernels = {
            'gaussian', 'tophat', 'epanechnikov',
            'exponential', 'linear', 'cosine'
        }

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Compute log‑density estimates for query points using a
        KernelDensity estimator.

        Parameters
        ----------
        problem : dict
            Must contain keys 'data_points', 'query_points', 'kernel',
            and 'bandwidth'.

        Returns
        -------
        dict
            ``{'log_density': [...]}`` on success
            ``{'error': '...'}`` on failure
        """
        try:
            # Fast conversion: no copy if already array
            X = np.asarray(problem['data_points'])
            X_q = np.asarray(problem['query_points'])
            kernel = problem['kernel']
            bandwidth = problem['bandwidth']

            # Basic shape checks
            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError('Data points or query points are not 2D arrays.')
            if X.shape[0] == 0:
                raise ValueError('No data points provided.')
            if X_q.shape[0] == 0:
                return {'log_density': []}
            if X.shape[1] != X_q.shape[1]:
                raise ValueError('Data points and query points have different dimensions.')

            # Validate bandwidth
            if not isinstance(bandwidth, Real) or bandwidth <= 0:
                raise ValueError('Bandwidth must be positive.')
            bandwidth = float(bandwidth)

            # Validate kernel
            if kernel not in self._kernels:
                raise ValueError(f'Unknown kernel: {kernel}')

            # Fit and score
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
            kde.fit(X)
            log_density = kde.score_samples(X_q)

            return {'log_density': log_density.tolist()}

        except KeyError as e:
            return {'error': f'Missing key: {e}'}
        except (ValueError, TypeError, NotFittedError, np.linalg.LinAlgError) as e:
            return {'error': f'Computation error: {e}'}
        except Exception as e:
            return {'error': f'Unexpected error: {e}'}