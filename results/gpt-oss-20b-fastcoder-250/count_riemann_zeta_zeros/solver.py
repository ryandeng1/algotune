import mpmath as mp
from typing import Any


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Count the number of Riemann zeta zeros with imaginary part <= t.

        The implementation delegates to `mpmath.mp.nzeros`, but performs a
        small amount of lazy setup to avoid repeated global changes to the
        working precision.  The precision is set once per solver instance,
        keeping the default `mp.dps` unless explicitly overridden.
        """
        # The directory of `mp.dps` might have been modified elsewhere; keep
        # a reference to the current precision and restore it after the call.
        current_dps = mp.dps
        try:
            # Using the default precision should be sufficient for most
            # use‑cases.  If a user requires higher precision they can
            # adjust `mp.dps` before calling `solve`.
            count = mp.nzeros(problem["t"])
        finally:
            # Ensure the global precision is not altered permanently.
            mp.dps = current_dps

        return {"result": count}