from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        # Pre‑process: build list of subsets seen as sets for quick lookup
        subsets = [set(s) for s in problem]
        m = len(subsets)

        # Build element coverage mapping once
        # assumes universe is {1..max_element}
        max_elem = max((max(s) for s in subsets if s), default=0)
        covers = [[] for _ in range(max_elem + 1)]
        for idx, subset in enumerate(subsets, 1):  # 1‑based literal index
            for e in subset:
                covers[e].append(idx)

        def sat_with_k(k: int) -> list[int] | None:
            cnf = CNF()
            # Coverage clauses
            for e in range(1, max_elem + 1):
                ce = covers[e]
                if ce:
                    cnf.append(ce)
                else:
                    # impossible to cover e
                    return None
            # Cardinality constraint
            lits = list(range(1, m + 1))
            cnf.extend(CardEnc.atmost(lits=lits, bound=k, encoding=EncType.seqcounter).clauses)
            # Solve
            with Solver(name="Minicard") as sol:
                sol.append_formula(cnf)
                if sol.solve():
                    mmodel = sol.get_model()
                    return [i for i in range(1, m + 1) if i in mmodel]
            return None

        # Binary search on solution size
        lo, hi = 1, m + 1
        best = None
        while lo < hi:
            mid = (lo + hi) // 2
            res = sat_with_k(mid)
            if res:
                best = res
                hi = mid
            else:
                lo = mid + 1
        return best or []