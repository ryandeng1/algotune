from typing import Any
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver


class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        try:
            n = len(problem)
            literals = [i + 1 for i in range(n)]
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if problem[i][j] == 1:
                        edges.append((i, j))
            
            # First check for k=0
            cnf = CNF()
            for i, j in edges:
                cnf.append([i + 1, j + 1])
            
            atmost_k = CardEnc.atmost(lits=literals, bound=0, encoding=EncType.seqcounter)
            cnf.extend(atmost_k.clauses)
            
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
                for i, j in edges:
                    cnf.append([i + 1, j + 1])
                
                atmost_k = CardEnc.atmost(lits=literals, bound=mid, encoding=EncType.seqcounter)
                cnf.extend(atmost_k.clauses)
                
                with Solver(name="Minicard") as solver:
                    solver.append_formula(cnf)
                    sat = solver.solve()
                    model = solver.get_model() if sat else None
                    if model:
                        selected = [i for i in range(n) if model[i] > 0]
                        right = len(selected)
                    else:
                        left = mid
            
            return selected
        except Exception as e:
            return list(range(len(problem)))