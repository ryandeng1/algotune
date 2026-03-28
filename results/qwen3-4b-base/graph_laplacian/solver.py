from typing import Any
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        try:
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]), shape=problem["shape"]
            )
            normed = problem["normed"]
        except Exception:
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [],
                    "shape": problem.get("shape", (0, 0)),
                }
            }

        try:
            L = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed)
            L_csr = L
        except Exception:
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [],
                    "shape": problem["shape"],
                }
            }

        return {
            "laplacian": {
                "data": L_csr.data.tolist(),
                "indices": L_csr.indices.tolist(),
                "indptr": L_csr.indptr.tolist(),
                "shape": L_csr.shape,
            }
        }