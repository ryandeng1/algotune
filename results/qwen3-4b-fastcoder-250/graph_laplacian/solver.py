import numpy as np
import scipy.sparse as sp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        data = problem["data"]
        indices = problem["indices"]
        indptr = problem["indptr"]
        shape = problem["shape"]
        normed = problem["normed"]
        n = shape[0]
        
        A = sp.csr_matrix((data, indices, indptr), shape=shape)
        
        if not normed:
            degrees = A.sum(axis=1).A1
            D = sp.diags(degrees, 0)
            L = D - A
        else:
            degrees = A.sum(axis=1).A1
            sqrt_degrees = np.sqrt(degrees)
            mask = sqrt_degrees == 0
            sqrt_degrees[mask] = 1e-10
            D_inv_sqrt = sp.diags(1.0 / sqrt_degrees, 0)
            L = sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
        
        L_csr = L.tocsr()
        return {
            "laplacian": {
                "data": L_csr.data.tolist(),
                "indices": L_csr.indices.tolist(),
                "indptr": L_csr.indptr.tolist(),
                "shape": L_csr.shape
            }
        }
