from typing import Any
import faiss
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points = np.array(problem["points"])
        queries = np.array(problem["queries"])
        k = problem["k"]
        dim = points.shape[1]

        index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(index)
        index.add_with_ids(points.astype(np.float32), np.arange(len(points)))

        k = min(k, len(points))
        distances, indices = index.search(queries.astype(np.float32), k)

        solution = {"indices": indices.tolist(), "distances": distances.tolist()}

        if problem.get("distribution") == "hypercube_shell":
            num_queries = 2 * dim
            bqs = np.zeros((num_queries, dim), dtype=np.float32)
            bqs[dim:] = 1.0
            bq_dist, bq_idx = index.search(bqs, k)
            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()

        return solution