from typing import Any
from ortools.sat.python import cp_model

class Solver:

    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the maximum clique problem using the CP‑SAT solver.

        :param problem: A 2D adjacency matrix representing the graph.
        :return: A list of node indices that form a maximum clique.
        """
        n = len(problem)
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f'x_{i}') for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 0:
                    model.Add(nodes[i] + nodes[j] <= 1)
                else:
                    pass
            else:
                pass
        else:
            pass
        model.Maximize(sum(nodes))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL:
            selected = [i for i in range(n) if solver.Value(nodes[i]) == 1]
            return selected
        else:
            return []
