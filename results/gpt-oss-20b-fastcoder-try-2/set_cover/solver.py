from typing import List
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver

class Solver:

    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solves the set cover problem using a SAT solver.

        The problem is given as a list of subsets.
        The task is to find the smallest subcollection of these subsets such that every
        element in the universe U (which is the union of all subsets and is assumed to be {1, 2, …, n})
        is covered.

        The returned indices are 1-indexed.

        :param problem: A list of subsets (each subset is a list of integers).
        :return: A list of indices (1-indexed) of the selected subsets.
        """
        m = len(problem)

        # pre‑compute element → covering subsets
        element_to_subsets = {}
        for idx, subset in enumerate(problem):
            for e in subset:
                element_to_subsets.setdefault(e, []).append(idx + 1)

        universe = list(element_to_subsets.keys())
        universe.sort()  # 1 .. n
        n = len(universe)

        def sat_for_k(k: int) -> List[int] | None:
            cnf = CNF()

            # coverage constraints
            for e in universe:
                lits = element_to_subsets[e]
                if lits:
                    cnf.append(lits)
                else:
                    cnf.append([1, -1])  # unsatisfiable clause

            # cardinality constraint: at most k subsets selected
            lits = list(range(1, m + 1))
            cnf.extend(CardEnc.atmost(lits=lits, bound=k, encoding=EncType.seqcounter).clauses)

            with Solver(name='m22', bootstrap_with=cnf.clauses) as slv:
                if slv.solve():
                    model = slv.get_model()
                    # keep only chosen subsets
                    return [i + 1 for i in range(m) if (i + 1) in model]
            return None

        # binary search for minimal k
        lo, hi = 1, m
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            res = sat_for_k(mid)
            if res is not None:
                best = res
                hi = mid - 1
            else:
                lo = mid + 1

        return best or []