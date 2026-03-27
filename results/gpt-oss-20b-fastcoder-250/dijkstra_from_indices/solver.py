import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any, List, Dict


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Solves all‑source shortest paths on a CSR graph using SciPy's
        Dijkstra implementation, converting infinite distances to None.
        The function is optimized for speed by relying on vectorised
        NumPy operations instead of Python loops.
        """
        try:
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            sources = problem["source_indices"]
            if not isinstance(sources, (list, np.ndarray)) or not sources:
                raise ValueError
        except Exception:
            return {"distances": []}

        try:
            # SciPy raises if directed or min_only absent; we provide defaults
            dist_matrix = scipy.sparse.csgraph.dijkstra(
                csgraph=graph_csr,
                directed=getattr(self, "directed", False),
                indices=sources,
                min_only=getattr(self, "min_only", False),
            )
        except Exception:
            return {"distances": []}

        # Convert the NumPy array to an object array in one shot
        obj_matrix = dist_matrix.astype(object)
        # Replace infinities with None
        np.where(np.isinf(obj_matrix), None, obj_matrix, out=obj_matrix)
        # Convert to a list of lists (or a single list if only one source)
        distances = obj_matrix.tolist()
        if isinstance(distances, list) and len(distances) == 1:
            distances = distances[0]
        return {"distances": distances}