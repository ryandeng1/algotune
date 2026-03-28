import numpy as np
import faiss
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Input arrays
        points_np = np.asarray(problem["points"], dtype=np.float32, order="C")
        queries_np = np.asarray(problem["queries"], dtype=np.float32, order="C")
        k = min(problem["k"], len(points_np))
        dim = points_np.shape[1]

        # Build index
        flat_index = faiss.IndexFlatL2(dim)
        flat_index = faiss.IndexIDMap(flat_index)
        ids = np.arange(len(points_np), dtype=np.int64)
        flat_index.add_with_ids(points_np, ids)

        # Main k‑nn search
        dist, idx = flat_index.search(queries_np, k)
        solution = {"indices": idx.tolist(), "distances": dist.tolist()}

        # Optional boundary search
        if problem.get("distribution") == "hypercube_shell":
            # Construct two boundary query points for each dimension
            bq = np.empty((2 * dim, dim), dtype=np.float32)
            # zeros
            bq[0::2] = 0.0
            # ones
            bq[1::2] = 1.0
            # set per-dimension differences
            for d in range(dim):
                bq[2 * d, d] = 0.0
                bq[2 * d + 1, d] = 1.0
            bd_dist, bd_idx = flat_index.search(bq, k)
            solution["boundary_distances"] = bd_dist.tolist()
            solution["boundary_indices"] = bd_idx.tolist()

        return solution