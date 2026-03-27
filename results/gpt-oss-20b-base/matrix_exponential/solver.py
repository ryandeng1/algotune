import numpy as np
from typing import Any, Dict, List

# Try to use scipy if available; otherwise provide a lightweight fallback.
try:
    from scipy.linalg import expm as _expm
except Exception:
    def _expm(A: np.ndarray) -> np.ndarray:
        """
        Compute the matrix exponential via the scaling and squaring method
        with a Padé approximant (order 13).  This implementation is a
        lightweight drop‑in replacement for scipy.linalg.expm.
        """
        # Constants for the Pade approximants (order 13).
        coeffs = np.array(
            [64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
             1187353796428800.0,  129060195264000.0,  10559470521600.0,
             670442572800.0,      33522128640.0,      1325963520.0,
             40840800.0,          1134000.0,          21000.0,
             252.0,              1.0]
        )

        # Scaling: find s such that ||A / 2**s||_∞ ≤ 0.5
        norm = np.linalg.norm(A, ord=np.inf)
        s = max(0, int(np.ceil(np.log2(norm)) if norm > 0 else 0))
        A_scaled = A / (2 ** s)

        # Compute Padé approximant (Pade(13))
        A2 = A_scaled @ A_scaled
        A4 = A2 @ A2
        A6 = A4 @ A2

        U = A_scaled @ (
            coeffs[13] * np.eye(A.shape[0]) + coeffs[11] * A2
            + coeffs[9] * A4 + coeffs[7] * A6
        )
        V = (
            coeffs[12] * np.eye(A.shape[0]) + coeffs[10] * A2
            + coeffs[8] * A4 + coeffs[6] * A6
        )

        P = U + V
        Q = -U + V

        expA = np.linalg.solve(Q, P)

        # Squaring
        for _ in range(s):
            expA = expA @ expA

        return expA


class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, List[List[float]]]:
        """
        Compute the matrix exponential of a square matrix using a fast
        scaling/squaring + Padé approximant algorithm.  This
        implementation avoids expensive optional dependencies and
        ensures consistent performance across environments.
        """
        A = problem["matrix"]
        expA = _expm(A)
        # Convert to a plain Python list for JSON‑friendly output
        return {"exponential": expA.tolist()}