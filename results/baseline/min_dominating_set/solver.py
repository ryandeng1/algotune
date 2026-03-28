from typing import Any
from ortools.sat.python import cp_model

class Solver:

    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the minimum dominating set problem using the CP-SAT solver.

        :param problem: A 2d adjacency matrix representing the graph.
        :return: A list of node indices included in the minimum dominating set.
        """
        n = len(problem)
        model = cp_model.CpModel()
        nodes = [model.NewBoolVar(f'x_{i}') for i in range(n)]
        for i in range(n):
            neighbors = [nodes[i]]
            for j in range(n):
                if problem[i][j] == 1:
                    neighbors.append(nodes[j])
                else:
                    pass
            else:
                pass
            model.Add(sum(neighbors) >= 1)
        else:
            pass
        model.Minimize(sum(nodes))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL:
            selected = [i for i in range(n) if solver.Value(nodes[i]) == 1]
            return selected
        else:
            return []
