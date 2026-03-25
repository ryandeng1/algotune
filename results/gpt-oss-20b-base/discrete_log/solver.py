# solver.py
import math
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, int], **kwargs) -> Dict[str, int]:
        """
        Solve g^x ≡ h (mod p) using the baby-step giant-step algorithm.

        Parameters
        ----------
        problem : dict
            Dictionary containing prime `p`, generator `g`, and target `h`.

        Returns
        -------
        dict
            Dictionary with key "x" containing the smallest non‑negative solution.
        """
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]

        # Baby-step giant-step
        n = math.isqrt(p) + 1

        # Precompute baby steps: g^j mod p
        baby = {}
        cur = 1
        for j in range(n):
            baby[cur] = j
            cur = (cur * g) % p

        # Compute g^-n mod p
        inv_g_n = pow(pow(g, n, p), p - 2, p)  # Fermat's little theorem

        cur = h
        for i in range(n):
            if cur in baby:
                return {"x": i * n + baby[cur]}
            cur = (cur * inv_g_n) % p

        # If no solution found, return -1 (should not happen for valid inputs)
        return {"x": -1}
