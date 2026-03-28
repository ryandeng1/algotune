import sys
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver

# -------------------------------------------------------------

def _build_cnf(adj, k):
    """Return CNF encoding of “find an independent set of size k”. """
    n = len(adj)
    cnf = CNF()
    # Edge constraints: no two adjacent vertices selected
    for i in range(n):
        ai = i + 1
        for j in range(i + 1, n):
            if adj[i][j]:
                cnf.append([ai, j + 1])
    # Cardinality constraint: at most k vertices can be selected
    atmost = CardEnc.atmost(lits=[i + 1 for i in range(n)], bound=k, encoding=EncType.seqcounter)
    cnf.extend(atmost.clauses)
    return cnf


def _find_mis(adj):
    """
    Binary search on cardinality with a SAT solver that more efficiently
    reuses the same solver instance between iterations.
    """
    n = len(adj)
    lo, hi = 0, n
    best_solution = []

    # Initial solver problems: set of clauses that never change
    base_cnf = CNF()
    for i in range(n):
        ai = i + 1
        for j in range(i + 1, n):
            if adj[i][j]:
                base_cnf.append([ai, j + 1])

    with Solver(name="Minicard") as solver:
        solver.append_formula(base_cnf)
        while lo < hi:
            mid = (lo + hi + 1) // 2  # try to build larger set
            solver.restart()  # clean auxiliary clauses but keep base
            atmost = CardEnc.atmost(lits=[i + 1 for i in range(n)], bound=mid, encoding=EncType.seqcounter)
            solver.append_formula(atmost.clauses)
            if solver.solve():
                model = solver.get_model()
                selected = [i for i, v in enumerate(model, start=1) if v > 0]
                best_solution = selected  # keep the best found so far
                lo = mid
            else:
                hi = mid - 1

    return [i for i in best_solution]


# -------------------------------------------------------------

class Solver:

    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solve the maximum independent set problem on an adjacency matrix.
        :param problem: adjacency matrix (list of lists)
        :return: list of 0‑based vertex indices included in a maximum independent set
        """
        return _find_mis(problem)


# -------------------------------------------------------------
if __name__ == "__main__":
    # Example usage: read adjacency matrix from stdin
    data = sys.stdin.read().strip().splitlines()
    if not data:
        sys.exit(0)
    n = int(data[0])
    mat = [list(map(int, line.split())) for line in data[1:1+n]]
    solver = Solver()
    res = solver.solve(mat)
    print(" ".join(map(str, res)))