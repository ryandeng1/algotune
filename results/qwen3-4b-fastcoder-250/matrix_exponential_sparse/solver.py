import math
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, sparse.spmatrix]) -> sparse.spmatrix:
        A = problem["matrix"]
        n = A.shape[0]
        
        result = sparse.eye(n, format='csc')
        current = sparse.eye(n, format='csc')
        
        for k in range(1, 21):
            current = A @ current
            result += current / math.factorial(k)
            if result.nnz > n * n - 100:
                break
        
        return result
