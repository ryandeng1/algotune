import numpy as np
import faiss
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert input points and queries to contiguous float32 arrays once
        points = np.asarray(problem["points"], dtype=np.float32)
        queries = np.asarray(problem["queries"], dtype=np.float32)
        k = min(problem["k"], len(points))
        dim = points.shape[1]

        # Build a flat L2 index (no precomputation needed)
        index = faiss.IndexFlatL2(dim)
        # Assign IDs so we can retrieve original indices
        index = faiss.IndexIDMap(index)
        index.add_with_ids(points, np.arange(len(points), dtype=np.int64))

        # Main k-NN search
        dists, idxs = index.search(queries, k)
        solution = {
            "indices": idxs.tolist(),
            "distances": dists.tolist()
        }

        # Optional boundary query handling
        if problem.get("distribution") == "hypercube_shell":
            # Prepare boundary queries efficiently: 2*dim vectors
            bqs = np.empty((2 * dim, dim), dtype=np.float32)
            for d in range(dim):
                bqs[2 * d,     d] = 0.0
                bqs[2 * d + 1, d] = 1.0
            bdists, bidxs = index.search(bqs, k)
            solution["boundary_distances"] = bdists.tolist()
            solution["boundary_indices"] = bidxs.tolist()

        return solution