from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver
from typing import List


class Solver:

    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solve the minimum set‑cover instance by a binary search on the number of
        chosen subsets and a single SAT query for each mid‑value.
        The SAT instance is built in a very cache‑friendly way to keep the
        critical path short.

        :param problem: list of subsets (each subset is a list of ints)
        :return: list of 1‑indexed subset indices that achieve the cover
        """
        # Pre‑process the input once: set representations and max element
        subsets = [set(s) for s in problem]
        max_elem = max((max(s) for s in subsets if s), default=0)
        m = len(problem)

        # Build a mapping from elements to the list of subsets that cover it
        cover_map: dict[int, List[int]] = {}
        for idx, subset in enumerate(subsets, start=1):
            for e in subset:
                cover_map.setdefault(e, []).append(idx)

        def build_cnf(k: int) -> CNF:
            """
            Return a CNF that enforces that each element
            1..max_elem is covered and that at most k subsets are chosen.
            """
            cnf = CNF()
            # coverage clauses
            for e in range(1, max_elem + 1):
                lits = cover_map.get(e)
                if lits:
                    cnf.append(lits)
                else:                # unsatisfiable if an element never appears
                    cnf.append([1, -1])

            # cardinality constraint – at most k
            top_lits = list(range(1, m + 1))
            atmost_k = CardEnc.atmost(lits=top_lits, bound=k, encoding=EncType.seqcounter)
            cnf.extend(atmost_k.clauses)
            return cnf

        left, right = 1, m
        best = None

        while left <= right:
            mid = (left + right) // 2
            cnf = build_cnf(mid)
            with Solver(name='Minicard') as solver:
                solver.append_formula(cnf)
                sat = solver.solve()
                model = solver.get_model() if sat else None
            if sat:
                selected = [idx for idx in range(1, m + 1) if idx in model]
                best = selected
                right = mid - 1
            else:
                left = mid + 1

        return best or []