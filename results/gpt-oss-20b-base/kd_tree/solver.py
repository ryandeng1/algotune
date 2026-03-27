from typing import Any
import numpy as np
import faiss


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert data to float32 once
        pts = np.asarray(problem["points"], dtype=np.float32)
        qrys = np.asarray(problem["queries"], dtype=np.float32)
        k = min(problem["k"], pts.shape[0])
        dim = pts.shape[1]

        # Build the FAISS index
        idx = faiss.IndexFlatL2(dim)
        idx = faiss.IndexIDMap(idx)
        idx.add_with_ids(pts, np.arange(pts.shape[0], dtype=np.int64))

        # Query the index
        dist, ind = idx.search(qrys, k)
        solution = {"indices": ind.tolist(), "distances": dist.tolist()}

        # Inject boundary queries for hypercube_shell distribution
        if problem.get("distribution") == "hypercube_shell":
            # Create all boundary queries at once
            bqs = np.zeros((2 * dim, dim), dtype=np.float32)
            bqs[:dim, :] = 1.0
            bqs[dim:, dim // 2] = 1.0  # set the differing dimension
            # Actually simpler: place 0 & 1 on each dimension
            for d in range(dim):
                bqs[d, d] = 0.0
                bqs[d + dim, d] = 1.0
            bd, bi = idx.search(bqs, k)
            solution["boundary_distances"] = bd.tolist()
            solution["boundary_indices"] = bi.tolist()

        return solution