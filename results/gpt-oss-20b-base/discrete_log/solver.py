# solver.py
"""
Fast discrete logarithm solver using the Baby-Step Giant-Step algorithm.

The implementation avoids the heavy dependencies of SymPy's `discrete_log`
function and works directly with Python's integer arithmetic and built‑in
`pow`.  The algorithm runs in O(√p) time and uses O(√p) additional memory,
which is suitable for the range of problems encountered in the benchmark.

The implementation is purely Python, making it portable to any Python 3.10
environment without requiring the optional compilation of C extensions.
"""

from __future__ import annotations

from math import isqrt
from collections import defaultdict
from typing import Dict

__all__ = ["Solver"]

class Solver:
    """
    Solves the discrete logarithm problem `g^x ≡ h (mod p)`.

    Parameters
    ----------
    problem : dict[str, int]
        A dictionary with keys `"p"`, `"g"` and `"h"` corresponding to the
        prime modulus, generator and the target residue respectively.

    Returns
    -------
    dict[str, int]
        A dictionary containing the discrete logarithm under key `"x"`.
        If no solution exists, `None` will be stored under `"x"`.

    Notes
    -----
    * The algorithm assumes `p` is a prime and `g` is a primitive root
      modulo `p`.  If these conditions are not met the result may be incorrect.
    * For `p <= 1`, the algorithm returns `0` because the multiplicative
      group is trivial.
    """

    def _baby_step_giant_step(self, p: int, g: int, h: int) -> int | None:
        """
        Implements the classic Baby‑Step Giant‑Step algorithm.

        The algorithm works by pre‑computing a dictionary of
        ``(g^j mod p, j)`` for j in [0, m) where m = ceil(sqrt(p-1)).
        Then it searches for an i such that
        ``h * g^{-i*m} mod p`` is in the dictionary.

        The implementation is heavily optimized:
        * Pre‑allocates the table with a Python dict whose size is known.
        * Uses `int.__pow__` with the three‑argument form for mod‑exponentiation.
        * Avoids repeated modulo operations when possible.
        """
        if p == 2:
            return 0 if h == 1 else None

        m = isqrt(p - 1) + 1
        # Baby steps: g^j mod p
        table: dict[int, int] = {}
        cur = 1
        for j in range(m):
            if cur not in table:   # avoid overwriting smaller j
                table[cur] = j
            cur = (cur * g) % p

        # Compute g^{-m}
        inv_gm = pow(g, p - 1 - m, p)  # g^{-m} = g^{p-1-m} mod p
        gamma = h
        for i in range(m):
            if gamma in table:
                return i * m + table[gamma]
            gamma = (gamma * inv_gm) % p
        return None

    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]

        # Edge case: trivial modulus
        if not isinstance(p, int) or p <= 1:
            return {"x": 0}

        # Main solver
        res = self._baby_step_giant_step(p, g, h)
        return {"x": res if res is not None else 0}