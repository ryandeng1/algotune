from typing import Any
import numpy as np
import faiss


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Prepare data
        points = np.asarray(problem["points"], dtype=np.float32)
        queries = np.asarray(problem["queries"], dtype=np.float32)
        k = min(problem["k"], len(points))
        dim = points.shape[1]

        # Build FAISS index
        index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(index)
        ids = np.arange(len(points), dtype=np.int64)
        index.add_with_ids(points, ids)

        # Main search
        dists, idxs = index.search(queries, k)

        # Prepare result
        solution = {"indices": [idxs.flatten().tolist()][:k], "distances": dists.tolist()}

        # Boundary queries if needed
        if problem.get("distribution") == "hypercube_shell":
            # Construct all boundary queries in one go
            bqs = []
            for d in range(dim):
                q0 = np.zeros(dim, dtype=np.float32)
                q1 = np.ones(dim, dtype=np.float32)
                q0[d] = 0.0
                q1[d] = 1.0
                bqs.append(q0)
                bqs.append(q1)
            bqs = np.stack(bqs, axis=0)
            bd, bi = index.search(bqs, k)
            solution["boundary_distances"] = bd.tolist()
            solution["boundary_indices"] = bi.tolist()

        return solution