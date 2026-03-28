import numpy as np
import faiss
from typing import Any, Dict


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Prepare data
        points = np.asarray(problem["points"], dtype=np.float32)
        queries = np.asarray(problem["queries"], dtype=np.float32)
        k = min(problem["k"], points.shape[0])
        dim = points.shape[1]

        # Build flat L2 index with ID mapping
        index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(index)
        index.add_with_ids(points, np.arange(points.shape[0], dtype=np.int64))

        # Search queries
        dists, idxs = index.search(queries, k)

        solution = {
            "indices": idxs.tolist(),
            "distances": dists.tolist(),
        }

        # Optional boundary query handling
        if problem.get("distribution") == "hypercube_shell":
            # Create 2*dim boundary queries (0/1 on each axis)
            bqs = np.zeros((2 * dim, dim), dtype=np.float32)
            bqs[:dim] = 0.0
            bqs[dim:] = 1.0
            bqs[np.arange(dim), np.arange(dim)] = 0.0
            bqs[np.arange(dim) + dim, np.arange(dim)] = 1.0

            bd, bi = index.search(bqs, k)
            solution["boundary_distances"] = bd.tolist()
            solution["boundary_indices"] = bi.tolist()

        return solution