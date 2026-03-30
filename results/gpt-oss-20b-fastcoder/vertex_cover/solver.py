# solver.py
from typing import List
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver

class Solver:
    """
    Fast implementation of maximum independent set via binary search + SAT.
    """
    def __init__(self) -> None:
        # Pre‑allocate minimal structs for speed
        self._edge_list: List[tuple[int, int]] = []

    def _build_edges(self, adj: List[List[int]]) -> None:
        """Fill _edge_list with all unordered edges (i < j)."""
        self._edge_list.clear()
        n = len(adj)
        for i in range(n):
            row = adj[i]
            for j in range(i + 1, n):
                if row[j]:
                    self._edge_list.append((i + 1, j + 1))

    def _sat_for_k(self, n: int, k: int) -> CNF:
        """
        Return a CNF encoding that enforces
        * every edge contains at least one chosen vertex
        * at most `k` vertices are chosen
        """
        cnf = CNF()
        # edge clauses (i ∨ j) for each edge
        cnf.extend([[u, v] for u, v in self._edge_list])
        # at most k selected
        cnf.extend(CardEnc.atmost(lits=list(range(1, n + 1)), bound=k,
                                  encoding=EncType.seqcounter).clauses)
        return cnf

    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Find a maximum independent set of the graph represented by the
        adjacency matrix `problem`. Returns the list of vertex indices
        (0‑based).
        """
        n = len(problem)
        if n == 0:
            return []

        # Build edge list once
        self._build_edges(problem)

        # Binary search on the cardinality of the maximum independent set
        lo, hi = 0, n
        best: List[int] = []

        # Cache solver instances for reuse (pysat does not allow re‑use of formula)
        while lo < hi:
            mid = (lo + hi) // 2
            # Construct CNF for current bound
            cnf = self._sat_for_k(n, mid)
            # Solve
            with Solver(name='Minicard', bootstrap_with=cnf.clauses) as solver:
                if solver.solve():
                    model = solver.get_model()
                    # model contains negated assignments too; pick positives
                    sat_set = [i for i, val in enumerate(model, start=1) if val > 0]
                    # We found an independent set of size <= mid
                    # The binary search is on *upper* bound, so shrink hi
                    hi = mid
                else:
                    # Need larger bound
                    lo = mid + 1

        # After binary search lo is the size of the maximum independent set
        # Build final CNF with exact bound lo
        cnf = self._sat_for_k(n, lo)
        with Solver(name='Minicard', bootstrap_with=cnf.clauses) as solver:
            solver.solve()  # guaranteed sat
            model = solver.get_model()
            best = [i for i, val in enumerate(model, start=1) if val > 0]

        # Convert to 0‑based indices
        return [idx - 1 for idx in best]