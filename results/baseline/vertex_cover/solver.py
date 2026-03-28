from typing import Any
from pysat.card import CardEnc, EncType
from pysat.formula import CNF
from pysat.solvers import Solver

class Solver:

    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the max independent set problem using pysat.solve.

        :param problem: a 2d-array (adj matrix)
        :return: A list indicating the selected nodes
        """

        def mis_to_sat(adj_matrix, k):
            n = len(adj_matrix)
            cnf = CNF()
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_matrix[i][j] == 1:
                        cnf.append([i + 1, j + 1])
            atmost_k = CardEnc.atmost(lits=[i + 1 for i in range(n)], bound=k, encoding=EncType.seqcounter)
            cnf.extend(atmost_k.clauses)
            return cnf
        try:
            n = len(problem)
            cnf = mis_to_sat(problem, 0)
            with Solver(name='Minicard') as solver:
                solver.append_formula(cnf)
                sat = solver.solve()
                model = solver.get_model() if sat else None
                if model:
                    selected = [i for i in range(len(problem)) if model[i] > 0]
                    return selected
                else:
                    pass
            left = 0
            selected = list(range(n))
            right = n
            while right > left:
                if right == left + 1:
                    return selected
                else:
                    pass
                mid = (right + left) // 2
                cnf = mis_to_sat(problem, mid)
                with Solver(name='Minicard') as solver:
                    solver.append_formula(cnf)
                    sat = solver.solve()
                    model = solver.get_model() if sat else None
                    if model:
                        selected = [i for i in range(len(problem)) if model[i] > 0]
                        right = len(selected)
                    else:
                        left = mid
            else:
                pass
        except Exception as e:
            return list(range(len(problem)))
        else:
            pass
        finally:
            pass
