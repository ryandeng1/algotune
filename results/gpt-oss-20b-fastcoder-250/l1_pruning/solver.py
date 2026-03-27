from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the l1‑pruning problem described in
        https://doi.org/10.1109/CVPR.2018.00890.

        The algorithm performs a projection of |v| onto
        the L1‑ball of radius k.  It runs in O(n log n) due to the
        single sort, and the rest uses vectorised NumPy operations
        for speed.

        Parameters
        ----------
        problem : dict
            Must contain keys "v" (iterable of numbers) and "k" (float).

        Returns
        -------
        dict
            {"solution": [float, ...]} – the projected vector with the
            original signs restored.
        """
        v = np.asarray(problem["v"], dtype=np.float64).ravel()
        k = float(problem["k"])

        # If the total L1 norm is already <= k, no pruning needed.
        l1_norm = np.abs(v).sum()
        if l1_norm <= k:
            return {"solution": v.tolist()}

        u = np.abs(v)

        # sort in descending order
        mu = np.sort(u)[::-1]

        # cumulative sum of mu
        cumul = np.cumsum(mu)

        # find the smallest j where mu[j] <= (cumul[j] - k) / (j + 1)
        # use broadcasting to avoid explicit loop
        j = np.where(mu <= (cumul - k) / np.arange(1, len(mu) + 1))[0]
        if j.size == 0:
            theta = (cumul[-1] - k) / len(mu)
        else:
            idx = j[0]
            theta = (cumul[idx] - k) / (idx + 1)

        w = np.maximum(u - theta, 0.0)

        # restore signs
        solution_vec = w * np.sign(v)

        return {"solution": solution_vec.tolist()}