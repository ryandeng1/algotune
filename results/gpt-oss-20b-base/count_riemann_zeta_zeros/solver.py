# solver.py

from typing import Any, Dict
from mpmath import mp

class Solver:
    """
    Computes the number of non‑trivial zeros of the Riemann zeta function
    with imaginary part ≤ t.  The computation is performed with
    ``mpmath.mp.nzeros`` which is already highly optimised in C.
    """

    # A single caller of ``solve`` will set ``mp.dps`` just once
    # to a value that gives sufficient precision for most test cases.
    _initialized = False

    def _ensure_initialized(self) -> None:
        """
        Initialise the mpmath context only once to avoid repeated
        re‑setting of ``mp.dps``.  This method is executed on the first
        call to ``solve``.
        """
        if not self._initialized:
            # Regular precision of 15 decimal places is usually enough
            # for the zero counting routine in typical benchmarks.
            mp.dps = max(15, mp.dps)
            self._initialized = True

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        problem
            Dictionary with key ``'t'`` which contains the height
            (imaginary part) up to which zeta zeros should be counted.

        Returns
        -------
        dict
            Dictionary containing the computed count under the key
            ``'result'``.
        """
        self._ensure_initialized()

        t = problem['t']
        # Use the highly optimised C implementation via mpmath
        result = mp.nzeros(t)
        return {'result': result}