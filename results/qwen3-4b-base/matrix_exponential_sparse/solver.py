from typing import Any
from scipy import sparse
from scipy.sparse.linalg import expm

class Solver:
    def solve(self, problem: dict[str, sparse.spmatrix]) -> sparse.spmatrix:
        A = problem["matrix"].tocsr()
        solution = expm(A)
        return solution