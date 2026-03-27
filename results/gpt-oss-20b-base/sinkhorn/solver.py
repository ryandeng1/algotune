from typing import Any, Dict, List, Union
import numpy as np
import ot


def solve(problem: Dict[str, Any]) -> Dict[str, Union[List[List[float]], None, str]]:
    """
    Compute the regularised optimal transport plan using POT's Sinkhorn algorithm.
    Parameters
    ----------
    problem : dict
        Must contain keys "source_weights", "target_weights", "cost_matrix" and "reg".
    Returns
    -------
    dict
        With keys:
            transport_plan : List[List[float]] | None
            error_message  : str | None
    """
    # Prepare inputs – avoid extra copies
    a = np.asarray(problem["source_weights"], dtype=np.float64, order="C")
    b = np.asarray(problem["target_weights"], dtype=np.float64, order="C")
    M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)

    # Ensure weight vectors sum to one (OT requires probability measures)
    if a.sum() == 0 or b.sum() == 0:
        return {"transport_plan": None, "error_message": "Zero‑mass source or target."}

    a = a / a.sum()
    b = b / b.sum()

    reg = float(problem["reg"])
    try:
        G = ot.sinkhorn(a, b, M, reg)
        if not np.isfinite(G).all():
            raise ValueError("Non‑finite values in transport plan")
        # Convert numpy array to nested Python lists for the expected return type
        plan = G.tolist()
        return {"transport_plan": plan, "error_message": None}
    except Exception as exc:                     # pragma: no cover – safety net
        return {"transport_plan": None, "error_message": str(exc)}