from typing import List
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver


class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve a set‑cover instance using binary search + a SAT solver.
        The input is a list of subsets (each a list of ints).  It is
        assumed that the universe is {1, 2, …, n} where n is the maximum
        element appearing in *problem*.  The result contains the 1‑based
        indices of the chosen subsets.
        """

        # ------------------------------------------------------------------
        #  Pre‑compute the universe and the list of subsets that contain each
        #  element so that we do not rebuild the same clauses on every
        #  iteration of the binary search.
        # ------------------------------------------------------------------
        universe_size = 0
        elem_to_subs: dict[int, List[int]] = {}

        for i, subset in enumerate(problem, start=1):
            for e in subset:
                if e > universe_size:
                    universe_size = e
                elem_to_subs.setdefault(e, []).append(i)

        # ------------------------------------------------------------------
        #  Helper that builds a CNF for a fixed bound k.
        # ------------------------------------------------------------------
        def build_cnf(k: int) -> CNF:
            cnf = CNF()

            # Coverage clauses.
            for e in range(1, universe_size + 1):
                covers = elem_to_subs.get(e)
                if covers:  # should always be true
                    cnf.append(covers)

            # Cardinality: at most k subsets are chosen.
            lits = list(range(1, len(problem) + 1))
            atmost = CardEnc.atmost(lits=lits, bound=k, encoding=EncType.seqcounter)
            cnf.extend(atmost.clauses)

            return cnf

        # ------------------------------------------------------------------
        #  Binary search on the minimum number of subsets required.
        # ------------------------------------------------------------------
        m = len(problem)
        lo, hi = 1, m            # the size of the solution is in [1, m]
        best: List[int] | None = None

        while lo <= hi:
            mid = (lo + hi) // 2
            cnf = build_cnf(mid)

            with Solver(name="minicard") as slv:
                slv.append_formula(cnf)
                sat = slv.solve()

                if sat:
                    model = slv.get_model()
                    # The model contains all clauses we've added.
                    # Positive numbers indicate the variable is True.
                    chosen = [i for i in model if i > 0 and i <= m]
                    best = chosen
                    hi = mid - 1        # try fewer subsets
                else:
                    lo = mid + 1        # need more subsets

        return best or []