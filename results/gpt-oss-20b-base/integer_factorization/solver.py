# solver.py

from typing import Dict
from sympy.ntheory import factorint

class Solver:
    """
    Fast factorization solver for 2‑prime composites.
    """

    def solve(self, problem: Dict[str, int]) -> Dict[str, int]:
        """
        Return the two prime factors of the composite number provided in ``problem``.
        The returned values are sorted such that p <= q.

        Parameters
        ----------
        problem : dict
            Must contain a key ``'composite'`` with an integer value.

        Returns
        -------
        dict
            ``{'p': int, 'q': int}`` – the two prime factors (p <= q).

        Raises
        ------
        ValueError
            If the input is not an integer or if the factorization does not
            yield exactly two prime factors (including the case p == q).
        """
        composite_val = problem.get('composite')
        if not isinstance(composite_val, int):
            raise ValueError(f"Expected integer composite, got {composite_val!r}")

        flt = factorint(composite_val)

        # Handle the normal case: two distinct primes.
        if len(flt) == 2 and all(exp == 1 for exp in flt.values()):
            p, q = sorted(flt.keys())
            return {'p': p, 'q': q}

        # Handle perfect square of a prime.
        if len(flt) == 1 and list(flt.values())[0] == 2:
            p = next(iter(flt))
            return {'p': p, 'q': p}

        raise ValueError(
            f"Expected exactly two prime factors, got {len(flt)} "
            f"(factorization: {flt})"
        )