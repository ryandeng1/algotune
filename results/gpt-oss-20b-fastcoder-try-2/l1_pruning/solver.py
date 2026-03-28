import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Implements the L1‑pruning projection from
        https://doi.org/10.1109/CVPR.2018.00890 in O(n log n) time.

        Parameters
        ----------
        problem : dict
            Must contain keys "v" (array-like) and "k" (non‑negative scalar).

        Returns
        -------
        dict
            ``{"solution": list}`` – the projected vector.
        """
        v = np.asarray(problem["v"]).astype(float)
        k = float(problem["k"])
        v = v.ravel()

        # magnitude ordering      
        abs_v = np.abs(v)
        order = np.argsort(abs_v)[::-1]                # descending by magnitude
        mu = abs_v[order]

        # cumulative sum of sorted magnitudes
        csum = np.cumsum(mu)

        # compute potential thresholds theta for each possible cut‑off
        idxs = np.arange(1, len(mu) + 1, dtype=float)
        theta = (csum - k) / idxs

        # find first index where mu[idx] > theta[idx]
        # (if none, use last index)
        mask = mu > theta
        if np.any(mask):
            j = mask.searchsorted(True)  # first True
        else:
            j = len(mu) - 1

        theta_opt = theta[j]

        # shrinkage
        w = np.maximum(abs_v - theta_opt, 0)
        # restore signs
        w *= np.sign(v)

        return {"solution": w.tolist()}