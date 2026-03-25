from typing import Any, Dict, List
import numpy as np
import faiss

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute k-nearest neighbors for each query point using faiss.
        The implementation mirrors the reference solution but is packaged
        solely for speed and correctness compliance.
        """
        # Extract inputs
        points = np.array(problem["points"], dtype=np.float32)
        queries = np.array(problem["queries"], dtype=np.float32)
        k = problem["k"]
        n_points = points.shape[0]

        # Prepare the index
        index = faiss.IndexFlatL2(points.shape[1])
        index = faiss.IndexIDMap(index)

        # Add points with explicit IDs
        ids = np.arange(n_points, dtype=np.int64)
        index.add_with_ids(points, ids)

        # Ensure k does not exceed number of points
        k = min(k, n_points)

        # Perform search
        distances, indices = index.search(queries, k)

        # Convert to Python lists
        result: Dict[str, Any] = {
            "indices": indices.tolist(),
            "distances": distances.tolist()
        }

        # Boundary queries for hypercube_shell distribution
        if problem.get("distribution") == "hypercube_shell":
            dim = points.shape[1]
            # Build 2*dim boundary queries: one per axis at 0 and 1
            boundary_queries = np.concatenate([
                np.full((dim, dim), 0.0, dtype=np.float32),
                np.eye(dim, dtype=np.float32)
            ])
            # Correct: we need one point with 0 at each axis and one with 1
            bqs = []
            for d in range(dim):
                q0 = np.zeros(dim, dtype=np.float32)
                q0[d] = 0.0
                q1 = np.ones(dim, dtype=np.float32)
                q1[d] = 1.0
                bqs.extend([q0, q1])
            bqs = np.stack(bqs, axis=0)

            bq_dist, bq_idx = index.search(bqs, k)
            result["boundary_distances"] = bq_dist.tolist()
            result["boundary_indices"] = bq_idx.tolist()

        return result
