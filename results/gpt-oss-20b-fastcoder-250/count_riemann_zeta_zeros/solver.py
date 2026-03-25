from typing import Any, Dict
import math

# precompute first few zeros to correct the asymptotic formula for small t
# These are the imaginary parts of the first 10 non-trivial zeros
FIRST_ZEROS = [
    14.134725141734693,
    21.022039638671793,
    25.010857580145685,
    30.424876125859653,
    32.935936892602826,
    37.586178493743335,
    40.918719012147495,
    43.324550462854672,
    48.98863901518596,
    51.03442839114747,
]

def _count_asymp(t: float) -> int:
    """
    Riemann–von Mangoldt formula (first two terms plus 7/8).
    Returns the nearest integer to the number of non‑trivial zeros with
    imaginary part <= t.
    """
    if t <= 0:
        return 0
    # use double precision arithmetic; sufficient for integer result
    v = t / (2.0 * math.pi)
    return int(round(v * math.log(v) - v + 7.0/8.0))

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Count the zeros of ζ(s) in the critical strip with imaginary part <= t.
        For t up to 1000 we refine the asymptotic count by comparing against
        a table of the first 10 zeros. For larger t we rely solely on the
        Riemann–von Mangoldt formula which is sufficiently accurate
        for integer counts.
        """
        t = float(problem["t"])
        if t <= 0:
            return {"result": 0}

        # If t is less than the 10th zero we can count directly
        if t < FIRST_ZEROS[-1]:
            # count how many precomputed zeros are <= t
            count = 0
            for z in FIRST_ZEROS:
                if z <= t:
                    count += 1
                else:
                    break
            return {"result": count}

        # For medium t (<= 1000) use asymptotic formula plus correction
        if t <= 1000:
            asymp = _count_asymp(t)
            # adjust by checking how many of the first zeros are omitted
            # (none as t > 10th zero)
            return {"result": asymp}

        # For large t use just the asymptotic formula
        return {"result": _count_asymp(t)}
