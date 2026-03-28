from typing import Any, Dict, List
import numpy as np
import faiss


class Solver:
    """Highly‑optimised nearest‑neighbour solver using FAISS."""

    def __init__(self) -> None:
        # No heavy initialisation needed; FAISS objects are cheap to create.
        pass

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Convert points and queries to float32 arrays only once
        points = problem["points"]
        queries = problem["queries"]
        k = int(problem["k"])
        points_f32 = np.asarray(points, dtype=np.float32)
        queries_f32 = np.asarray(queries, dtype=np.float32)
        n_points, dim = points_f32.shape

        # Build a flat L2 index and map ids
        index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(index)
        index.add_with_ids(points_f32, np.arange(n_points, dtype=np.int64))

        # Ensure k does not exceed available points
        k = min(k, n_points)

        # Main search: return distances and indices
        dists, idxs = index.search(queries_f32, k)

        result: Dict[str, Any] = {
            "indices": idxs.tolist(),
            "distances": dists.tolist(),
        }

        # Optional boundary queries for hypercube_shell distribution
        if problem.get("distribution") == "hypercube_shell":
            # Create all boundary query vectors in a single array
            # Each dimension contributes two queries: all 0's and all 1's
            base_vectors = np.ones((dim, dim), dtype=np.float32)

            # 0 vector for each dimension (explicit)
            zeros_vec = np.zeros((dim, dim), dtype=np.float32)

            # Flip the d-th coordinate to 0 for each query
            for d in range(dim):
                zeros_vec[d, d] = 0.0

            # Stack them: [q0_0, q1_0, q0_1, q1_1, ...]
            boundary_qs = np.vstack([zeros_vec, base_vectors])

            b_dist, b_idx = index.search(boundary_qs, k)
            result["boundary_distances"] = b_dist.tolist()
            result["boundary_indices"] = b_idx.tolist()

        return result