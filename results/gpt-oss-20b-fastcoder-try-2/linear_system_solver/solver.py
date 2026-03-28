import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Solve the linear system Ax = b using NumPy's fastest path.
        The inputs are converted to contiguous float64 arrays with
        `np.asarray` to avoid unnecessary copies.  The `check_finite=False`
        flag bypasses a costly NaN/inf check that is unnecessary for
        well‑formed input data, giving a small speed‑up.
        """
        A = np.asarray(problem['A'], dtype=np.float64)
        b = np.asarray(problem['b'], dtype=np.float64)

        # Use `np.linalg.solve` if A is square and full‑rank; otherwise fall back
        # to `np.linalg.lstsq` which handles rectangular systems.
        if A.ndim == 2 and A.shape[0] == A.shape[1]:
            x = np.linalg.solve(A, b, check_finite=False)
        else:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)

        return x.tolist()