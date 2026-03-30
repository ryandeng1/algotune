# solver.py
from typing import List

from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver as SATSolver


class Solver:

    # ---------------------------------------------------------------------
    # Helper: build the compulsory part of the CNF (the coverage constraints).
    # It does not depend on the cardinality bound `k` and can therefore be
    # computed once and re‑used for every iteration of the binary search.
    # ---------------------------------------------------------------------
    def _build_coverage_cnf(self, subsets: List[List[int]]) -> List[List[int]]:
        """Return a list of clauses ensuring every element is covered."""
        # Universe is assumed to be {1, …, n}.  Collect it once.
        universe = set()
        for s in subsets:
            universe.update(s)

        clauses = []
        for elem in range(1, len(universe) + 1):
            # list of subset indices (1‑based) that contain `elem`
            covers = [i + 1 for i, s in enumerate(subsets) if elem in s]
            # If no subset contains the element the formula is unsatisfiable,
            # expressed by the empty clause [1, -1] which makes the solver fail.
            clauses.append(covers if covers else [1, -1])
        return clauses

    # ---------------------------------------------------------------------
    # Main solve routine.
    # ---------------------------------------------------------------------
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solves the set‑cover problem using a SAT encoding.

        :param problem: List of subsets (each subset is a list of ints).
        :return: List of 1‑based indices of the selected subsets.
        """
        m = len(problem)
        if m == 0:
            return []

        # Pre‑compute the purely coverage part of the CNF.
        coverage_clauses = self._build_coverage_cnf(problem)

        left, right = 1, m + 1           # binary search on the cardinality
        best_solution: List[int] | None = None

        while left < right:
            mid = (left + right) // 2

            # Build a copy of the base CNF and add the cardinality constraint.
            cnf = CNF()
            cnf.extend(coverage_clauses)

            lits = list(range(1, m + 1))  # 1‑based variables for the subsets
            atmost_k = CardEnc.atmost(lits=lits, bound=mid, encoding=EncType.seqcounter)
            cnf.extend(atmost_k.clauses)

            with SATSolver(name='Minicard') as solver:
                solver.append_formula(cnf)
                if solver.solve():
                    # `model` contains both positive and negative literals.
                    # We only keep the positive ones: the selected subsets.
                    model = solver.get_model()
                    selected = [i + 1 for i in range(m) if i + 1 in model]
                    best_solution = selected
                    right = len(selected)
                else:
                    left = mid + 1

        return best_solution or []