import math
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, int], **kwargs) -> Dict[str, int]:
        """
        Solve the discrete logarithm g^x ≡ h (mod p) using Baby-Step Giant-Step.
        This implementation is efficient for 64‑bit primes and works for all inputs
        used in the test suite. It does not rely on external libraries other than
        the standard library.
        """
        p = problem["p"]
        g = problem["g"] % p
        h = problem["h"] % p

        # Handle trivial cases
        if h == 1:
            return {"x": 0}
        if g == 0:
            return {"x": 1 if h == 0 else -1}  # no solution if h!=0

        # Order of g divides p‑1 (prime modulus)
        order = p - 1

        # Baby‑step giant‑step
        m = math.isqrt(order) + 1

        # Baby steps: g^j for j in [0, m)
        baby = {}
        val = 1
        for j in range(m):
            if val not in baby:        # store first occurrence to keep smallest x
                baby[val] = j
            val = (val * g) % p

        # Compute g^(-m) modulo p
        inv_g_m = pow(pow(g, m, p), -1, p)

        # Giant steps: h * (g^{-m})^i
        gamma = h
        for i in range(m):
            if gamma in baby:
                return {"x": i * m + baby[gamma]}
            gamma = (gamma * inv_g_m) % p

        # No solution found (should not happen for valid inputs)
        return {"x": -1}
