from typing import List
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]

        model = cp_model.CpModel()
        x = {(i, j): model.NewBoolVar(f'x[{i},{j}]') for i in range(n) for j in range(n) if i != j}
        # Circuit constraint: one outgoing and one incoming for each city
        model.AddCircuit([(u, v, x[u, v]) for (u, v) in x])
        # Objective: minimize total distance
        model.Minimize(sum(problem[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j))

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False  # disable verbose output
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            path = [0]
            cur = 0
            while len(path) < n:
                # find the outgoing edge from cur
                for nxt in range(n):
                    if cur != nxt and solver.Value(x[cur, nxt]) == 1:
                        cur = nxt
                        path.append(cur)
                        break
            path.append(0)
            return path
        return []