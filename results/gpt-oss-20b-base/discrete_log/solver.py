import math
from typing import Dict

class Solver:
    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        """
        Solve the discrete logarithm problem g^x ≡ h (mod p)
        using the Baby‑Step Giant‑Step algorithm.
        The function returns a dictionary with the key "x" set to the
        smallest non‑negative solution.
        """
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]

        # Trivial cases
        if h % p == 1:
            return {"x": 0}
        if g % p == 0:
            # g == 0 modulo p only works if h == 0
            return {"x": 1 if h % p == 0 else -1}

        # Find m = ceil(sqrt(p-1)) for the algorithm
        m = math.isqrt(p - 1) + 1

        # Baby steps: store g^j for j in [0, m-1]
        baby = {}
        cur = 1
        for j in range(m):
            if cur not in baby:
                baby[cur] = j
            cur = (cur * g) % p

        # Compute g^{-m} modulo p
        inv_g_m = pow(g, (p - 1) - (m % (p - 1)), p)

        # Giant steps
        gamma = h % p
        for i in range(m):
            if gamma in baby:
                x = i * m + baby[gamma]
                return {"x": x}
            gamma = (gamma * inv_g_m) % p

        # No solution found (should not happen for prime modulus with primitive root)
        return {"x": -1}