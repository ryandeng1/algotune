# solver.py
import numpy as np
import faiss
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Convert data to numpy arrays
        points = np.array(problem["points"], dtype=np.float32)
        queries = np.array(problem["queries"], dtype=np.float32)
        k = int(problem["k"])
        n_points = points.shape[0]

        # Build FAISS index (FlatL2) and add points with IDs
        dim = points.shape[1]
        index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(index)
        id_array = np.arange(n_points, dtype=np.int64)
        index.add_with_ids(points.astype(np.float32), id_array)

        # Clamp k if necessary
        k = min(k, n_points)

        # Perform search
        distances, indices = index.search(queries.astype(np.float32), k)

        # Prepare result
        result = {
            "indices": indices.tolist(),
            "distances": distances.tolist(),
        }

        # Boundary checks for hypercube_shell distribution (optional)
        if problem.get("distribution") == "hypercube_shell":
            bqs = []
            for d in range(dim):
                q0 = np.zeros(dim, dtype=np.float32)
                q1 = np.ones(dim, dtype=np.float32)
                bqs.append(q0)
                bqs.append(q1)
            bqs = np.vstack(bqs)
            bq_dist, bq_idx = index.search(bqs.astype(np.float32), k)
            result["boundary_distances"] = bq_dist.tolist()
            result["boundary_indices"] = bq_idx.tolist()

        return result
