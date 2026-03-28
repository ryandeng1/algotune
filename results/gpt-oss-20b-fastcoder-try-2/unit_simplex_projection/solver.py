import numpy as np

class Solver:
    def solve(self, problem: dict[str, object]) -> dict[str, list]:
        """
        Solve the projection onto the probability simplex in O(n log n).
        """
        # Extract and flatten the vector
        y = np.asarray(problem.get('y'), dtype=np.float64).ravel()

        # Quick exit for trivial cases
        n = y.size
        if n == 0:
            return {'solution': np.array([], dtype=np.float64)}
        if np.all(y >= 0) and np.isclose(y.sum(), 1.0):
            return {'solution': y.copy()}

        # Sort in descending order
        y_sorted = np.sort(y)[::-1]

        # Prefix sums minus 1
        cum = np.cumsum(y_sorted) - 1.0

        # Find the largest rho where sorted_y > cum / 1..n
        idx = np.arange(1, n + 1, dtype=np.int64)
        mask = y_sorted > cum / idx
        rho = np.nonzero(mask)[0][-1]  # guaranteed at least one

        # Compute theta
        theta = cum[rho] / (rho + 1)

        # Projection
        x = np.maximum(y - theta, 0.0)

        return {'solution': x}