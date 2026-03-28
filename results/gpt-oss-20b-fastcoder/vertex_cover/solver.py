from typing import List
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver


class Solver:
    """
    A fast SAT‑based implementation for the Maximum Independent Set (MIS) problem.
    The solver is tuned for small to medium sized graphs (n <= ~100). For larger
    graphs the SAT engine still works but the run‑time may grow quickly.
    """

    def _mis_to_sat(self, adj: List[List[int]], k: int) -> CNF:
        """
        Convert the MIS problem with an upper bound `k` on the size of the solution
        into a CNF formula. The resulting CNF contains:
          1. For each edge (i, j) a clause (¬v_i ∨ ¬v_j) forbidding both endpoints
             to be chosen simultaneously.
          2. An at‑most‑k cardinality constraint on the `n` variables.
        """
        n = len(adj)
        cnf = CNF()

        # Edge clauses: ¬v_i ∨ ¬v_j  ⇒  (i+1) ∨ (j+1) in CNF format
        for i in range(n):
            row = adj[i]
            for j, val in enumerate(row[i + 1 :], start=i + 1):
                if val:  # edge exists
                    cnf.append([i + 1, j + 1])  # both positive -> forbids both 1

        # Cardinality constraint: at most `k` variables can be true
        lits = [i + 1 for i in range(n)]
        atmost = CardEnc.atmost(lits=lits, bound=k, encoding=EncType.seqcounter)
        cnf.extend(atmost.clauses)
        return cnf

    def solve(self, problem: List[List[int]]) -> List[int]:
        """Return a list of node indices that form a maximum independent set."""
        n = len(problem)

        # Quick black‑box 0‑size check to handle empty graph
        empty_cnf = self._mis_to_sat(problem, 0)
        with Solver(name="Minicard") as s:
            s.append_formula(empty_cnf)
            if s.solve():
                model = s.get_model()
                return [i for i, v in enumerate(model) if v > 0]

        # Binary search over possible independent set sizes
        lo, hi = 0, n
        best = []
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cnf = self._mis_to_sat(problem, mid)
            with Solver(name="Minicard") as s:
                s.append_formula(cnf)
                sat = s.solve()
                if sat:
                    model = s.get_model()
                    cur = [i for i, v in enumerate(model) if v > 0]
                    best = cur
                    lo = mid
                else:
                    hi = mid - 1

        return best