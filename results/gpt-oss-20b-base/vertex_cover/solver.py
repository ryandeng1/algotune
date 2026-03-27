from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver
from typing import List


class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the maximum independent set problem via a SAT formulation.
        The function returns a list of vertex indices (0‑based) that
        constitute an independent set of maximum size.
        """
        n = len(problem)

        # Flatten all adjacency clauses once
        base_clauses: List[List[int]] = []
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j] == 1:
                    base_clauses.append([i + 1, j + 1])  # x_i or x_j

        def build_cnf(min_size: int) -> CNF:
            """Return a CNF for a given  minimum independent set size."""
            cnf = CNF()
            cnf.extend(base_clauses)

            # cardinality constraint: at least  min_size literals are true
            #  use at-most encoding of the complement
            atmost = CardEnc.atmost(
                lits=list(range(1, n + 1)),
                bound=n - min_size,
                encoding=EncType.seqcounter,
            )
            cnf.extend(atmost.clauses)
            return cnf

        # Quick check: can we pick all vertices?  (should always be false except empty graph)
        with Solver(name="MiniCard") as sol:
            sol.append_formula(build_cnf(n))
            if sol.solve():
                model = sol.get_model()
                return [i for i in range(n) if model[i] > 0]

        # Binary search for the maximum size
        low, high = 0, n
        best_set: List[int] = []

        while low < high:
            mid = (low + high + 1) // 2
            with Solver(name="MiniCard") as sol:
                sol.append_formula(build_cnf(mid))
                if sol.solve():
                    model = sol.get_model()
                    best_set = [i for i in range(n) if model[i] > 0]
                    low = mid
                else:
                    high = mid - 1

        return best_set