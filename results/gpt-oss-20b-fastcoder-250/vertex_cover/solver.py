import sys
from pyparsing import pp

from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver as SATSolver


# Helper to build the SAT encoding in a single pass
def _mis_to_sat(adj, k):
    n = len(adj)
    cnf = CNF()

    # independence constraints: x_i ∨ x_j for each edge
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if ai[j]:
                cnf.append([i + 1, j + 1])

    # cardinality constraint: at most n − k vertices not in the set
    # (minimizing the complement is a bit simpler for PySAT)
    atmost = CardEnc.atmost(lits=[i + 1 for i in range(n)], bound=n - k, encoding=EncType.seqcounter)
    cnf.extend(atmost.clauses)
    return cnf


def solve(problem: list[list[int]]) -> list[int]:
    """
    Compute a maximum independent set for an undirected graph given as an adjacency matrix.
    Returns the list of zero‑based vertex indices that form the set.
    """
    n = len(problem)

    # Special case: empty graph – all vertices can be taken
    if all(not any(row) for row in problem):
        return list(range(n))

    # Binary search for the maximum cardinality
    lo, hi = 0, n
    best_set = list(range(n))
    while lo <= hi:
        mid = (lo + hi) // 2
        cnf = _mis_to_sat(problem, mid)
        with SATSolver(name="minicard") as solver:
            solver.append_formula(cnf)
            sat = solver.solve()
            if sat:
                # Extract model and keep only indices of positive literals
                model = solver.get_model()
                selected = [i for i, lit in enumerate(model[:n], start=1) if lit > 0]
                # Found a set of size ≥ mid; try to increase
                best_set = selected
                lo = mid + 1
            else:
                # Not possible → shrink the target size
                hi = mid - 1
    return best_set