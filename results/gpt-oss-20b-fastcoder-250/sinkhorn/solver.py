import numpy as np
import ot
from typing import Any, Dict, List, Union

def solve(problem: Dict[str, Any]) -> Dict[str, Union[List[List[float]], None, str]]:
    # Convert weights and cost matrix to contiguous float64 arrays (without unnecessary copies)
    a = np.asarray(problem["source_weights"], dtype=np.float64, order="C")
    b = np.asarray(problem["target_weights"], dtype=np.float64, order="C")
    M = np.asarray(problem["cost_matrix"], dtype=np.float64, order="C")

    reg = float(problem["reg"])

    try:
        # Compute the Sinkhorn transport plan
        G = ot.sinkhorn(a, b, M, reg, log=False)

        # Quick sanity check for finite values
        if not np.all(np.isfinite(G)):
            raise ValueError("Non‑finite values in transport plan")

        # Convert array to nested Python lists for the expected output format
        return {"transport_plan": G.tolist(), "error_message": None}

    except Exception as exc:   # pragma: no cover
        # Return the error string instead of the plan
        return {"transport_plan": None, "error_message": str(exc)}