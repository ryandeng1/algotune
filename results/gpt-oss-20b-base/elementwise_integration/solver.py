#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from math import gamma, factorial, log

# We avoid importing heavy integrators; instead we evaluate the series definition of
# the integral analytically.  For each i:
#   ∫ Φ(a,b;x) dx = Σ_{k=0}^∞ (u^{k+1}-l^{k+1}) / ((k+1) k! Γ(ak+b))
# We truncate when the term becomes smaller than a tight relative tolerance.

# Small helper to compute a single integral value.
def _integral_term(a: float, b: float, l: float, u: float,
                   tol: float = 1e-14,
                   max_iter: int = 1000) -> float:
    res = 0.0
    k = 0
    # Precompute the denominator part that does not depend on k
    while k < max_iter:
        denom = (k + 1) * factorial(k) * gamma(a * k + b)
        if denom == 0:
            break
        term = (u ** (k + 1) - l ** (k + 1)) / denom
        res += term
        # Break when term relative to result is below tolerance or term is tiny
        if abs(term) < abs(res) * tol or abs(term) < tol:
            break
        k += 1
    return res

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        a_list: List[float] = problem["a"]
        b_list: List[float] = problem["b"]
        lower_list: List[float] = problem["lower"]
        upper_list: List[float] = problem["upper"]

        n = len(a_list)
        out = np.empty(n, dtype=np.float64)

        # Use vectorized loop for simplicity
        for i in range(n):
            out[i] = _integral_term(a_list[i], b_list[i],
                                    lower_list[i], upper_list[i])

        return {"result": out.tolist()}
