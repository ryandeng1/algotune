import numpy as np
from sklearn.neighbors import KernelDensity
from numbers import Real
from sklearn.exceptions import NotFittedError

class Solver:
    def __init__(self):
        self.available_kernels = {
            'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'
        }

    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        try:
            X = np.array(problem['data_points'])
            X_q = np.array(problem['query_points'])
            kernel = problem['kernel']
            bandwidth = problem['bandwidth']

            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError('Data points or query points are not 2D arrays.')
            if X.size == 0:
                raise ValueError('No data points provided.')
            if X_q.size == 0:
                return {'log_density': []}
            if X.shape[1] != X_q.shape[1]:
                raise ValueError('Data points and query points have different dimensions.')

            if not isinstance(bandwidth, Real) or bandwidth <= 0:
                raise ValueError('Bandwidth must be positive.')
            bandwidth = float(bandwidth)
            if kernel not in self.available_kernels:
                raise ValueError(f'Unknown kernel: {kernel}')

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