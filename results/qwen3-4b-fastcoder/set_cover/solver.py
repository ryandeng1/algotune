from typing import Any
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        universe = set()
        for subset in problem:
            universe.update(subset)
        n = len(universe)
        
        coverage = [[] for _ in range(n + 1)]
        for idx, subset in enumerate(problem):
            for e in subset:
                coverage[e].append(idx)
        
        m = len(problem)
        left = 1
        right = m + 1
        best_solution = None

        while left < right:
            mid = (left + right) // 2
            cnf = CNF()
            
            for e in range(1, n + 1):
                literals = [idx + 1 for idx in coverage[e]]
                if literals:
                    cnf.append(literals)
                else:
                    cnf.append([1, -1])
            
            lits = [i + 1 for i in range(m)]
            atmost_k = CardEnc.atmost(lits=lits, bound=mid, encoding=EncType.seqcounter)
            cnf.extend(atmost_k.clauses)
            
            with Solver(name="Minicard") as solver:
                solver.append_formula(cnf)
                sat = solver.solve()
                model = solver.get_model() if sat else None
            if sat and model is not None:
                selected = [i + 1 for i in range(m) if (i + 1) in model]
                best_solution = selected
                right = mid
            else:
                left = mid + 1
        
        return best_solution if best_solution is not None else []