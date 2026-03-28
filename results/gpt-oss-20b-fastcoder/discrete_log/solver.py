#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Dict

class Solver:
    """Fast Baby‑Step Giant‑Step discrete‑log implementation."""

    def _bsgs(self, p: int, g: int, h: int) -> int:
        """Return x such that g^x ≡ h (mod p). Assumes p is prime and g is a primitive root."""
        n = int(math.isqrt(p - 1)) + 1

        # Baby steps: g^j (mod p), 0 <= j < n
        table: Dict[int, int] = {}
        cur = 1
        for j in range(n):
            if cur not in table:          # keep smallest j for correctness
                table[cur] = j
            cur = (cur * g) % p

        # Precompute g^-n (mod p)
        inv_g_n = pow(g, p - 1 - n, p)     # Fermat: g^(p-1) ≡ 1

        cur = h
        for i in range(n):
            if cur in table:
                return i * n + table[cur]
            cur = (cur * inv_g_n) % p

        raise ValueError("Discrete logarithm not found")
    
    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        """Solve the discrete logarithm problem using an efficient BSGS algorithm."""
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        return {"x": self._bsgs(p, g, h)}