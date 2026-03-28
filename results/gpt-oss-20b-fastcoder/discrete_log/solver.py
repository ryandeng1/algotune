from math import isqrt
from typing import Dict, Any

# A lightweight baby‐step giant‐step implementation
def discrete_log_bs(p: int, g: int, h: int) -> int:
    """
    Solve for x in g^x ≡ h (mod p) using the baby‑step giant‑step algorithm.
    Works for prime p and g a primitive root modulo p.
    """
    if h == 1:
        return 0

    m = isqrt(p) + 1

    # Baby steps: g^j for j in [0, m)
    baby = {}
    cur = 1
    for j in range(m):
        if cur not in baby:          # keep the smallest j
            baby[cur] = j
        cur = (cur * g) % p

    # Pre‑compute g^(-m) mod p
    inv_g_m = pow(g, p - 1 - m, p)   # g^{-m} ≡ g^{p-1-m} (mod p)
    gamma = h
    for i in range(m):
        if gamma in baby:
            return i * m + baby[gamma]
        gamma = (gamma * inv_g_m) % p

    raise ValueError("Discrete log not found")

class Solver:

    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        """
        Solve discrete logarithm using a fast baby‑step giant‑step implementation.
        The function assumes that the input values are integers and that p is prime.
        """
        p = problem['p']
        g = problem['g']
        h = problem['h']
        return {'x': discrete_log_bs(p, g, h)}