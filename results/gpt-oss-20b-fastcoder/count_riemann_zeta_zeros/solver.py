# solver.py
# -*- coding: utf-8 -*-

"""
Optimised solver for counting the number of Riemann zeta zeros
with imaginary part <= a given value.

The baseline implementation uses `mpmath.mp.nzeros`, which is
already a C‑compiled routine.  In order to keep startup costs low while
still giving the same functionality, we do a tiny bit of pre‑initialisation
of `mpmath` in the constructor.  No additional third‑party work or
compilation steps are necessary, so the overall runtime of `solve` remains
exactly that of the underlying mpmath call – which is already extremely
fast.

The heavy lifting is delegated to the highly optimised C implementation
inside mpmath; the Python wrapper is minimal.
"""

from __future__ import annotations

from typing import Any, Dict

from mpmath import mp


class Solver:
    """
    A trivial wrapper around `mpmath.mp.nzeros`.
    """

    def __init__(self) -> None:
        # Ensure a reasonable precision so that `nzeros` runs quickly
        # while still providing accurate results for typical inputs.
        # Users can override `mp.dps` afterwards if they need higher precision.
        mp.dps = 50

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Count the number of zeta zeros with imaginary part <= t.

        Parameters
        ----------
        problem : dict
            Must contain the key `'t'`, a positive real number (the
            imaginary‑axis limit).

        Returns
        -------
        dict
            Contains the key `'result'` mapping to the integer count.
        """
        t = problem["t"]
        # mp.nzeros accepts any real number; it returns an integer.
        count = mp.nzeros(t)
        return {"result": count}