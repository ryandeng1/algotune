from typing import Any
import faiss
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points = np.asarray(problem["points"], dtype=np.float32)
        queries = np.asarray(problem["queries"], dtype=np.float32)
        k = min(problem["k"], len(points))
        dim = points.shape[1]

        # Build a flat L2 index
        index = faiss.IndexFlatL2(dim)
        index.add(points)

        # Main search
        dists, idxs = index.search(queries, k)

        solution = {
            "indices": idxs.tolist(),
            "distances": dists.tolist()
        }

        # Boundary queries – only for hypercube_shell distribution
        if problem.get("distribution") == "hypercube_shell":
            # 0‑vectors and 1‑vectors for every dimension
            bqs = np.empty((2 * dim, dim), dtype=np.float32)
            # first dim rows are all zeros
            # next dim rows are all ones
            bqs[dim:] = 1.0

            bq_d, bq_i = index.search(bqs, k)
            solution.update(
                {
                    "boundary_distances": bq_d.tolist(),
                    "boundary_indices": bq_i.tolist(),
                }
            )
        return solution