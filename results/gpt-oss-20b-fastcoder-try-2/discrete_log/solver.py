from math import isqrt
from typing import Dict

# ------------------------------------------------------------------------------
# A fast baby‑step giant‑step implementation for discrete logarithms
# ------------------------------------------------------------------------------

def _baby_step_giant_step(g: int, h: int, p: int) -> int:
    """
    Computes x such that g**x ≡ h (mod p) using the baby‑step giant‑step algorithm.
    Assumes that a solution exists and that all values are positive integers.
    """
    m = isqrt(p) + 1
    # Baby step:  g^j for j in [0, m)
    tbl: dict[int, int] = {}
    val = 1
    for j in range(m):
        tbl[val] = j
        val = (val * g) % p

    # Pre‑compute g^{-m} mod p
    inv_g = pow(g, -1, p)
    factor = pow(inv_g, m, p)

    # Giant step
    gamma = h
    for i in range(m):
        if gamma in tbl:
            return i * m + tbl[gamma]
        gamma = (gamma * factor) % p

    raise ValueError("No discrete logarithm found")

# ------------------------------------------------------------------------------
# Solver
# ------------------------------------------------------------------------------

class Solver:
    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        """
        Solves the discrete logarithm problem using an efficient
        baby‑step giant‑step algorithm.

        Parameters
        ----------
        problem: dict
            Dictionary containing the keys:
                - 'p': modulus (prime)
                - 'g': base (generator)
                - 'h': target value

        Returns
        -------
        dict
            Dictionary containing the key 'x' with the discrete logarithm.
        """
        p = problem['p']
        g = problem['g']
        h = problem['h']
        return {'x': _baby_step_giant_step(g, h, p)}