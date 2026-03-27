from typing import Any, List, Dict
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        try:
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            sources = problem["source_indices"]
            if not isinstance(sources, list) or not sources:
                raise ValueError
        except Exception:
            return {"distances": []}

        try:
            dist = scipy.sparse.csgraph.dijkstra(
                csgraph=graph_csr,
                directed=self.directed,
                indices=sources,
                min_only=self.min_only,
            )
        except Exception:
            return {"distances": []}

        # Replace infinities with None
        if dist.ndim == 1:
            out = [[None if np.isinf(d) else d for d in dist]]
        else:
            out = [[None if np.isinf(d) else d for d in row] for row in dist]

        return {"distances": out}