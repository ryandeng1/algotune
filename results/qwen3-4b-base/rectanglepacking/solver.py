import itertools
import math
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        W, H, rects = problem
        n = len(rects)
        w = [r[0] for r in rects]
        h = [r[1] for r in rects]
        rotatable = [r[2] for r in rects]
        
        model = cp_model.CpModel()
        
        x1_vars = [model.NewIntVar(0, W, f"x1_{i}") for i in range(n)]
        y1_vars = [model.NewIntVar(0, H, f"y1_{i}") for i in range(n)]
        rotated_vars = [model.NewBoolVar(f"rotated_{i}") for i in range(n)]
        placed_vars = [model.NewBoolVar(f"placed_{i}") for i in range(n)]
        
        for i in range(n):
            if not rotatable[i]:
                model.Add(rotated_vars[i] == 0)
        
        for i in range(n):
            model.Add(
                placed_vars[i] * (x1_vars[i] + w[i] * (1 - rotated_vars[i]) + h[i] * rotated_vars[i]) <= W
            )
            model.Add(
                placed_vars[i] * (y1_vars[i] + h[i] * (1 - rotated_vars[i]) + w[i] * rotated_vars[i]) <= H
            )
            model.Add(x1_vars[i] == 0).OnlyEnforceIf(placed_vars[i].Not())
            model.Add(y1_vars[i] == 0).OnlyEnforceIf(placed_vars[i].Not())
        
        for i in range(n):
            for j in range(i + 1, n):
                w_i, h_i = w[i], h[i]
                w_j, h_j = w[j], h[j]
                
                left_of_j = model.NewBoolVar(f"i_left_of_j_{i}_{j}")
                right_of_j = model.NewBoolVar(f"i_right_of_j_{i}_{j}")
                below_j = model.NewBoolVar(f"i_below_j_{i}_{j}")
                above_j = model.NewBoolVar(f"i_above_j_{i}_{j}")
                
                model.Add(
                    x1_vars[i] + w_i * (1 - rotated_vars[i]) + h_i * rotated_vars[i] <= x1_vars[j]
                ).OnlyEnforceIf([placed_vars[i], placed_vars[j], left_of_j])
                
                model.Add(
                    x1_vars[j] + w_j * (1 - rotated_vars[j]) + h_j * rotated_vars[j] <= x1_vars[i]
                ).OnlyEnforceIf([placed_vars[i], placed_vars[j], right_of_j])
                
                model.Add(
                    y1_vars[i] + h_i * (1 - rotated_vars[i]) + w_i * rotated_vars[i] <= y1_vars[j]
                ).OnlyEnforceIf([placed_vars[i], placed_vars[j], below_j])
                
                model.Add(
                    y1_vars[j] + h_j * (1 - rotated_vars[j]) + w_j * rotated_vars[j] <= y1_vars[i]
                ).OnlyEnforceIf([placed_vars[i], placed_vars[j], above_j])
                
                model.Add(left_of_j + right_of_j + below_j + above_j >= 1).OnlyEnforceIf([placed_vars[i], placed_vars[j]])
        
        model.Maximize(sum(placed_vars))
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 900.0
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = []
            for i in range(n):
                if solver.Value(placed_vars[i]):
                    x = solver.Value(x1_vars[i])
                    y = solver.Value(y1_vars[i])
                    rotated = solver.Value(rotated_vars[i]) == 1
                    solution.append((i, x, y, rotated))
            return solution
        else:
            return []
