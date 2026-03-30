# solver.py
import faiss
import numpy as np
from typing import Any, Dict, List


class Solver:
    """
    Fast nearest‑neighbour solver using Faiss.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # Convert input lists to contiguous float32 arrays (Faiss only accepts 32‑bit floats)
        # ------------------------------------------------------------------
        points = np.asarray(problem["points"], dtype=np.float32)
        queries = np.asarray(problem["queries"], dtype=np.float32)

        n_points, dim = points.shape
        k = min(int(problem["k"]), n_points)

        # ------------------------------------------------------------------
        # Build the Faiss index once per call
        # ------------------------------------------------------------------
        index: faiss.Index = faiss.IndexFlatL2(dim)          # L2 distance
        index = faiss.IndexIDMap(index)                       # keep IDs consistent
        index.add_with_ids(points, np.arange(n_points, dtype=np.int64))

        # ------------------------------------------------------------------
        # Main batch search
        # ------------------------------------------------------------------
        dists, idxs = index.search(queries, k)

        # ------------------------------------------------------------------
        # Pack the answer
        # ------------------------------------------------------------------
        solution: Dict[str, List[List[int]]] = {
            "indices": idxs.tolist(),
            "distances": dists.tolist(),
        }

        # ------------------------------------------------------------------
        # Optional boundary queries – only when needed
        # ------------------------------------------------------------------
        if problem.get("distribution") == "hypercube_shell":
            # Build the two extreme points per dimension
            #  [0,0,...,0], [1,1,...,1] with a single coordinate set to 1.0 or 0.0
            bqs = np.empty((2 * dim, dim), dtype=np.float32)
            for d in range(dim):
                bqs[2 * d, d] = 0.0
                bqs[2 * d + 1, d] = 1.0

            bdists, bidxs = index.search(bqs, k)
            solution["boundary_distances"] = bdists.tolist()
            solution["boundary_indices"] = bidxs.tolist()

        return solution