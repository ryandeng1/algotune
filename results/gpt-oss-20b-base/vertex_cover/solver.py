from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver

def solve(problem: list[list[int]]) -> list[int]:
    """
    Solves the maximum independent set problem using a SAT solver.
    :param problem: adjacency matrix (list of lists of 0/1)
    :return: list of vertex indices (0‑based) that form a maximum independent set
    """
    n = len(problem)
    # Pre‑compute the conflict clauses: -i \/ -j for each edge (i<j)
    conflict_clauses = []
    for i in range(n):
        row = problem[i]
        for j in range(i + 1, n):
            if row[j]:
                conflict_clauses.append([- (i + 1), -(j + 1)])

    # Binary search on the independent set size
    low, high = 0, n
    best_model = None
    while low <= high:
        mid = (low + high) // 2
        cnf = CNF()
        cnf.extend(conflict_clauses)
        # add cardinality constraint: at most mid variables can be true
        cnf.extend(CardEnc.atmost(lits=[i + 1 for i in range(n)],
                                  bound=mid,
                                  encoding=EncType.seqcounter).clauses)
        with Solver(name='glucose3', bootstrap_with=cnf) as solver:
            if solver.solve():
                best_model = solver.get_model()
                low = mid + 1
            else:
                high = mid - 1

    if best_model is None:
        return []
    return [i for i, v in enumerate(best_model[:n]) if v > 0]