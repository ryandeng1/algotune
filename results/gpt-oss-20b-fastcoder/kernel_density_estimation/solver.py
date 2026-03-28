import numpy as np
from sklearn.neighbors import KernelDensity
from numbers import Real

class Solver:
    available_kernels = [
        'gaussian', 'tophat', 'epanechnikov',
        'exponential', 'linear', 'cosine'
    ]

    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        try:
            X = np.asarray(problem['data_points'])
            Xq = np.asarray(problem['query_points'])
            kernel = problem['kernel']
            bandwidth = problem['bandwidth']

            if X.ndim != 2 or Xq.ndim != 2:
                raise ValueError('Data points or query points must be 2‑D arrays.')
            if X.size == 0:
                raise ValueError('No data points provided.')
            if Xq.shape[0] == 0:
                return {'log_density': []}
            if X.shape[1] != Xq.shape[1]:
                raise ValueError('Input and query dimensions differ.')
            if not isinstance(bandwidth, Real) or bandwidth <= 0:
                raise ValueError('Bandwidth must be a positive real number.')

            if kernel not in self.available_kernels:
                raise ValueError(f'Unknown kernel: {kernel}')

            kde = KernelDensity(kernel=kernel, bandwidth=float(bandwidth))
            kde.fit(X)
            return {'log_density': kde.score_samples(Xq).tolist()}

        except KeyError as e:
            return {'error': f'Missing key: {e}'}
        except Exception as e:
            return {'error': f'Computation error: {e}'}