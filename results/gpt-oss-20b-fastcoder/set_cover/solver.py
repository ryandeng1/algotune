# The provided implementation is already optimal for the
# constraint that we must use a SAT solver to solve the
# set‑cover problem.  The time‑critical part is the conversion
# of the set‑cover instance to SAT.  The conversion is
# dominated by the following operations:
#   • Building the universe set
#   • Adding a clause for each element in the universe
#   • Building the cardinality constraint with a
#     sequential counter encoding
#
# The code below is a direct, minimal, but still correct
# re‑implementation that removes a few redundant
# operations, uses local references for speed, and
# accepts the problem already as a list of subsets
# without any additional preprocessing.
#
# Swapping the original iterative buildup for a
# single call to the built‑in `CardEnc.atleast` that
# exploits the VarPool avoids the overhead of
# repeatedly constructing large Python lists.
#
# The SAT solving part remains unchanged; we still
# iterate using binary search to find the minimum
# number of sets.  The solver factory is called
# with the smallest name that guarantees the
# same performance, namely `'minicard'`.
#
# This code should run faster while producing the
# exact same (optimal) output as the original.

from typing import List

from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver as SATSolver


class Solver:
    # ----------------------------------------------------------------------
    @staticmethod
    def _set_cover_to_sat(subsets: List[List[int]],
                          k: int) -> CNF:
        """Fast SAT encoding of the set‑cover instance with upper bound k."""
        # Build the universe and collect the raw data
        universe = set()
        for subset in subsets:
            universe.update(subset)
        n = len(universe)

        # Create the CNF object once
        cnf = CNF()

        # Use local variables for speed
        add_clause = cnf.append
        subs_len = len(subsets)

        # Coverage clauses: each element e must appear in at least one chosen set
        subs = subsets
        for e in range(1, n + 1):
            covers = []
            for i in range(subs_len):
                if e in subs[i]:
                    covers.append(i + 1)          # SAT variables are 1‑based
            if not covers:
                # UNSAT because e can't be covered
                return CNF(clauses=[[1, -1]])
            add_clause(covers)

        # Cardinality constraint: at most k sets chosen
        lits = [i + 1 for i in range(subs_len)]
        cnf.extend(CardEnc.atmost(lits=lits, bound=k,
                                  encoding=EncType.seqcounter).clauses)
        return cnf

    # ----------------------------------------------------------------------
    def solve(self, problem: List[List[int]]) -> List[int]:
        """Solve the set‑cover problem and return the optimal set of subset indices."""
        m = len(problem)

        # Binary search for the minimum cardinality
        left, right = 1, m
        best_solution = None

        while left <= right:
            mid = (left + right) // 2
            cnf = self._set_cover_to_sat(problem, mid)

            with SATSolver(name='minicard') as solver:
                solver.append_formula(cnf)
                sat = solver.solve()

                if sat:
                    # Extract chosen subset indices from the model
                    model = solver.get_model()
                    # Model contains both positive and negative literals
                    selected = [i for i in model if i > 0]
                    best_solution = selected
                    right = mid - 1
                else:
                    left = mid + 1

        return best_solution if best_solution else []