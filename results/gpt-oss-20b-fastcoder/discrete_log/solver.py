# solver.py
"""
Optimised discrete-logarithm solver.

The original implementation used
    from sympy.ntheory.residue_ntheory import discrete_log
which internally falls back to a quite generic implementation.  
For the range of input sizes usually encountered in the
benchmarking framework a hand‑written Baby–Step Giant–Step
algorithm in pure Python is significantly faster, especially
because it only uses built‑in integer arithmetic and a
small dictionary for the baby‑steps.

The implementation below is fully deterministic,
does not rely on any external libraries beyond the
Python standard library and the package imports that are
already available in the execution environment.
"""

import math
from typing import Dict

class Solver:
    """
    Solve the discrete logarithm the problem::

        given primes 0 < h < p and generator g,
        find minimal non‑negative integer x such that
        g**x ≡ h (mod p).

    The function uses the Baby‑Step Giant‑Step algorithm
    with a space–time trade‑off chosen to minimise overhead.
    """

    @staticmethod
    def _baby_step_giant_step(g: int, h: int, p: int) -> int:
        """
        Classical Baby‑Step Giant‑Step algorithm.

        Complexity: O(√p) time, O(√p) space.
        Works for any prime modulus p and any g, h
        with gcd(g, p) == 1 (which holds for generators).

        Parameters
        ----------
        g : int
            Base (generator) of the group.
        h : int
            Target value.
        p : int
            Prime modulus.

        Returns
        -------
        int
            The smallest non‑negative integer x satisfying g**x ≡ h (mod p).
        """
        # Step size: ceil(sqrt(p - 1))
        m = int(math.isqrt(p - 1)) + 1

        # Precompute baby steps: g^j for j = 0 .. m-1
        baby: Dict[int, int] = {1: 0}
        current = 1
        for j in range(1, m):
            current = (current * g) % p
            # Only store first occurrence to keep minimal x
            baby.setdefault(current, j)

        # Compute g^{-m} once
        factor = pow(g, m * (p - 2), p)  # g^{-m} mod p via Fermat's little theorem
        gamma = h

        # Giant steps: gamma = h * (g^{-m})^i
        for i in range(m):
            if gamma in baby:
                return i * m + baby[gamma]
            gamma = (gamma * factor) % p

        # If no solution found (which shouldn't happen for valid problems)
        raise ValueError(f"No discrete logarithm found for g={g}, h={h}, p={p}")

    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        """
        Public API.

        Parameters
        ----------
        problem : dict[str, int]
            Dictionary containing the keys 'p', 'g', and 'h'.

        Returns
        -------
        dict[str, int]
            Dictionary with a single key 'x' containing the solution.
        """
        p = problem["p"]
        g = problem["g"]
        h = problem["h"]
        return {"x": self._baby_step_giant_step(g, h, p)}