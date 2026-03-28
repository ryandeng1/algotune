from typing import Any
from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]

        model = cp_model.CpModel()

        x = {(i, j): model.NewBoolVar(f"x[{i},{j}]") for i in range(n) for j in range(n) if i != j}

        model.AddCircuit([(u, v, var) for (u, v), var in x.items()])

        model.Minimize(sum(problem[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j))

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            next_city = [-1] * n
            for i in range(n):
                for j in range(n):
                    if i != j and solver.Value(x[i, j]) == 1:
                        next_city[i] = j
                        break
            path = [0]
            current = 0
            for _ in range(n - 1):
                current = next_city[current]
                path.append(current)
            path.append(0)
            return path
        else:
            return []