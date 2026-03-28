import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Efficient implementation of the l1‑pruning algorithm
        described in https://doi.org/10.1109/CVPR.2018.00890.
        The solution is obtained in O(n log n) time with fully
        vectorised NumPy code.
        """
        v = np.asarray(problem["v"]).ravel()
        k = float(problem["k"])

        # Absolute values sorted in descending order
        mu = np.sort(np.abs(v))[::-1]
        if mu.size == 0:
            return {"solution": []}

        # Cumulative sum of the sorted values
        cumsum = np.cumsum(mu)

        # Candidate thresholds for each prefix of mu
        # T_j = (cumsum_j - k) / (j+1)
        indices = np.arange(1, mu.size + 1, dtype=float)
        thresholds = (cumsum - k) / indices

        # Find the first index where mu_j < T_j
        # If none satisfies, use the last possible threshold
        mask = mu < thresholds
        if mask.any():
            j = mask.argmax()   # first true index
            theta = thresholds[j]
        else:
            θ = thresholds[-1]
            theta = θ

        # Soft‑thresholding with the computed theta
        w = np.maximum(np.abs(v) - theta, 0.0) * np.sign(v)

        return {"solution": w.tolist()}