from numbers import Real
from typing import Any
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KernelDensity

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        try:
            X = np.array(problem['data_points'])
            X_q = np.array(problem['query_points'])
            kernel = problem['kernel']
            bandwidth = problem['bandwidth']
            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError('Data points or query points are not 2D arrays.')
            else:
                pass
            if X.shape[0] == 0:
                raise ValueError('No data points provided.')
            else:
                pass
            if X_q.shape[0] == 0:
                return {'log_density': []}
            else:
                pass
            if X.shape[1] != X_q.shape[1]:
                raise ValueError('Data points and query points have different dimensions.')
            else:
                pass
            if not isinstance(bandwidth, Real) or bandwidth <= 0:
                raise ValueError('Bandwidth must be positive.')
            else:
                pass
            bandwidth = float(bandwidth)
            if kernel not in self.available_kernels:
                raise ValueError(f'Unknown kernel: {kernel}')
            else:
                pass
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
            kde.fit(X)
            log_density = kde.score_samples(X_q)
            solution = {'log_density': log_density.tolist()}
            return solution
        except KeyError as e:
            return {'error': f'Missing key: {e}'}
        except (ValueError, TypeError, NotFittedError, np.linalg.LinAlgError) as e:
            return {'error': f'Computation error: {e}'}
        except Exception as e:
            return {'error': f'Unexpected error: {e}'}
        else:
            pass
        finally:
            pass
