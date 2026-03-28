from typing import Any
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver


class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        edge_clauses = []
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    edge_clauses.append([i + 1, j + 1])
        
        try:
            cnf = CNF()
            cnf.extend(edge_clauses)
            atmost_0 = CardEnc.atmost(lits=[i + 1 for i in range(n)], bound=0, encoding=EncType.seqcounter)
            cnf.extend(atmost_0.clauses)
            
            with Solver(name="Minicard") as solver:
                solver.append_formula(cnf)
                sat = solver.solve()
                model = solver.get_model() if sat else None
                if model:
                    selected = [i for i in range(n) if model[i] > 0]
                    return selected
            
            left = 0
            selected = list(range(n))
            right = n
            
            while right > left:
                if right == left + 1:
                    return selected
                mid = (right + left) // 2
                cnf = CNF()
                cnf.extend(edge_clauses)
                atmost_mid = CardEnc.atmost(lits=[i + 1 for i in range(n)], bound=mid, encoding=EncType.seqcounter)
                cnf.extend(atmost_mid.clauses)
                
                with Solver(name="Minicard") as solver:
                    solver.append_formula(cnf)
                    sat = solver.solve()
                    model = solver.get_model() if sat else None
                    if model:
                        selected = [i for i in range(n) if model[i] > 0]
                        right = len(selected)
                    else:
                        left = mid
        except Exception:
            return list(range(n))