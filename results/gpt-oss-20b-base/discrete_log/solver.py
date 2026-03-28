import math
from collections import defaultdict

class Solver:
    """Fast baby‑step giant‑step discrete logarithm solver."""

    def solve(self, problem: dict[str, int]) -> dict[str, int]:
        """Return x such that g**x ≡ h (mod p)."""
        p, g, h = problem["p"], problem["g"], problem["h"]

        # Special cases
        if h % p == 1:
            return {"x": 0}
        if g % p == 0:
            return {"x": 1}  # only possible when h ≡ 0 (mod p)

        m = int(math.isqrt(p - 1)) + 1

        # Baby steps: g^j mod p, j = 0 .. m-1
        baby = {}
        cur = 1
        for j in range(m):
            if cur not in baby:          # keep the smallest exponent
                baby[cur] = j
            cur = (cur * g) % p

        # Compute g^{-m} mod p
        inv_g = pow(g, p - 2, p)          # Fermat's little theorem
        factor = pow(inv_g, m, p)

        cur = h
        for i in range(m):
            if cur in baby:
                return {"x": i * m + baby[cur]}
            cur = (cur * factor) % p

        # No solution found
        return {"x": None}