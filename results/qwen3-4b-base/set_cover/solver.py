from typing import Any
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver


class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        m = len(problem)
        universe = set()
        for subset in problem:
            universe.update(subset)
        n = len(universe)
        coverage = {e: [] for e in range(1, n + 1)}
        for i, subset in enumerate(problem):
            for e in subset:
                if e in coverage:
                    coverage[e].append(i + 1)
        
        left = 1
        right = m + 1
        best_solution = None
        
        while left < right:
            mid = (left + right) // 2
            cnf = CNF()
            for e in range(1, n + 1):
                if coverage[e]:
                    cnf.append(coverage[e])
                else:
                    cnf.append([1, -1])
            
            lits = [i + 1 for i in range(m)]
            atmost_k = CardEnc.atmost(lits=lits, bound=mid, encoding=EncType.seqcounter)
            cnf.extend(atmost_k.clauses)
            
            with Solver(name="Minicard") as solver:
                solver.append_formula(cnf)
                sat = solver.solve()
                if sat:
                    model = solver.get_model()
                    selected = [i + 1 for i in range(m) if (i + 1) in model]
                    best_solution = selected
                    right = len(selected)
                else:
                    left = mid + 1
        
        return best_solution or []