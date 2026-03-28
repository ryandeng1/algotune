import numpy as np

class Solver:
    """
    Optimised 1‑D Wasserstein distance calculator for equally spaced support.
    The support is assumed to be [1, 2, ..., n].
    """

    def solve(self, problem: dict[str, list[float]]) -> float:
        """
        Compute the 1‑D Wasserstein distance between two discrete distributions
        defined on the same equally spaced support.

        :param problem:  Dictionary with keys 'u' and 'v', each a list of positive
                        weights that sum to 1 (normalised probability mass).
        :return: Float distance.
        """
        u = np.asarray(problem["u"], dtype=np.float64)
        v = np.asarray(problem["v"], dtype=np.float64)

        # Defensive copy in case inputs are not already normalised.
        # This normalisation is inexpensive compared to other operations.
        sum_u = u.sum()
        if sum_u != 0.0:
            u /= sum_u
        sum_v = v.sum()
        if sum_v != 0.0:
            v /= sum_v

        # Compute cumulative distribution functions
        cu = np.cumsum(u)
        cv = np.cumsum(v)

        # L1 distance between the two CDFs is the 1‑D Wasserstein distance
        return np.abs(cu - cv).sum()