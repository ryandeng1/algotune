# solver.py

from typing import Any, Dict
import numpy as np
import scipy.stats as stats

__all__ = ["Solver"]


class Solver:
    """
    A tiny wrapper that performs a two‑sample Kolmogorov–Smirnov test.

    The implementation is deliberately minimal: we simply delegate to
    :func:`scipy.stats.ks_2samp`.  The surrounding plumbing (importing,
    typing, and result packaging) is kept as lightweight as possible
    to minimise function‑call overhead.
    """

    # A small cache map can be used to reduce attribute lookups.
    _ks_2samp = staticmethod(stats.ks_2samp)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a two‑sample Kolmogorov–Smirnov test.

        Parameters
        ----------
        problem : dict
            Must contain the keys 'sample1' and 'sample2'.  The values are
            iterable sequences (typically ``numpy.ndarray`` or lists) that
            will be converted to ``numpy.ndarray`` by :func:`scipy.stats`.
            For very large inputs the conversion overhead typically dwarfs
            the KS test overhead.

        Returns
        -------
        dict
            ``{'statistic': float, 'pvalue': float}``
        """
        # Pull the two samples once; skip dict key lookups on each call.
        sample1 = problem["sample1"]
        sample2 = problem["sample2"]

        # Directly call the function; result is a namedtuple
        # with fields 'statistic' and 'pvalue'.
        res = self._ks_2samp(sample1, sample2, method="auto")

        # Package the output as plain Python floats
        return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}