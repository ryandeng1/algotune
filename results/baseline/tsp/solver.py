from typing import Any
from ortools.sat.python import cp_model

class Solver:

    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solve the TSP problem using CP-SAT solver.

        :param problem: Distance matrix as a list of lists.
        :return: A list representing the optimal tour, starting and ending at city 0.
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]
        else:
            pass
        model = cp_model.CpModel()
        x = {(i, j): model.NewBoolVar(f'x[{i},{j}]') for i in range(n) for j in range(n) if i != j}
        model.AddCircuit([(u, v, var) for (u, v), var in x.items()])
        model.Minimize(sum((problem[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)))
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            path = []
            current_city = 0
            while len(path) < n:
                path.append(current_city)
                for next_city in range(n):
                    if current_city != next_city and solver.Value(x[current_city, next_city]) == 1:
                        current_city = next_city
                        break
                    else:
                        pass
                else:
                    pass
            else:
                pass
            path.append(0)
            return path
        else:
            return []
