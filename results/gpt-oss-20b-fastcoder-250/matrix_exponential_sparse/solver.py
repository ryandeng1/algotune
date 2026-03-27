import scipy.sparse.linalg

class Solver:
    def solve(self, problem: dict[str, scipy.sparse.spmatrix]) -> scipy.sparse.spmatrix:
        return scipy.sparse.linalg.expm(problem["matrix"])