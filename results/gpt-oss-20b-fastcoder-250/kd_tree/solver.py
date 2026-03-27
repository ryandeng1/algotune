from typing import Any
import numpy as np
import faiss


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Prepare data in a single dtype conversion
        points = problem["points"].astype(np.float32, copy=False)
        queries = problem["queries"].astype(np.float32, copy=False)
        k = min(problem["k"], len(points))
        dim = points.shape[1]

        # Build the index once
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        index.add_with_ids(points, np.arange(len(points), dtype=np.int64))

        # Main search
        distances, indices = index.search(queries, k)

        solution = {
            "indices": indices.tolist(),
            "distances": distances.tolist()
        }

        # Boundary queries for hypercube_shell tests
        if problem.get("distribution") == "hypercube_shell":
            # Construct all‑zero and all‑one rows with only the d‑th coordinate set
            eye = np.eye(dim, dtype=np.float32)
            zeros = eye.copy()          # zeros on main diagonal, ones elsewhere
            ones = eye.copy()           # ones on main diagonal, zeros elsewhere
            zeros[..., :] = 0.0
            ones[..., :] = 1.0
            zeros[np.arange(dim), np.arange(dim)] = 0.0
            ones[np.arange(dim), np.arange(dim)] = 1.0
            bqs = np.concatenate([zeros, ones], axis=0)

            bq_dist, bq_idx = index.search(bqs, k)
            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()

        return solution