from __future__ import annotations
from typing import Any, Dict, List, Union

import numpy as np
import ot


def solve(problem: dict[str, Any]) -> Dict[str, Union[List[List[float]], None, str]]:
    """
    Computes the regularized optimal transport plan between the source and target
    distributions using the Sinkhorn algorithm from POT.
    """
    try:
        # Convert inputs to NumPy arrays only once
        a = np.array(problem["source_weights"], dtype=np.float64, copy=False)
        b = np.array(problem["target_weights"], dtype=np.float64, copy=False)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)

        # Run Sinkhorn
        G = ot.sinkhorn(a, b, M, float(problem["reg"]))

        if not np.isfinite(G).all():
            raise ValueError("Non‑finite values in transport plan")

        # Convert to plain Python lists for the expected return type
        return {"transport_plan": G.tolist(), "error_message": None}

    except Exception as exc:  # pragma: no cover
        return {"transport_plan": None, "error_message": str(exc)}