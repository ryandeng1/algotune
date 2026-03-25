import itertools
from typing import List, Tuple, Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: Tuple[int, int, List[Tuple[int, int, bool]]], **kwargs) -> List[Tuple[int, int, int, bool]]:
        W, H, rectangles = problem
        n = len(rectangles)
        
        model = cp_model.CpModel()
        
        # Create variables
        placed = [model.NewBoolVar(f'placed_{i}') for i in range(n)]
        x = [model.NewIntVar(0, W, f'x_{i}') for i in range(n)]
        y = [model.NewIntVar(0, H, f'y_{i}') for i in range(n)]
        rotated = [model.NewBoolVar(f'rotated_{i}') for i in range(n)]
        
        # Add dimension constraints
        for i in range(n):
            w, h, rotatable = rectangles[i]
            model.Add(
                x[i] + (w if not rotated[i] else h) <= W
            ).OnlyEnforceIf(placed[i])
            model.Add(
                y[i] + (h if not rotated[i] else w) <= H
            ).OnlyEnforceIf(placed[i])
        
        # Add non-overlapping constraints
        for i in range(n):
            for j in range(i + 1, n):
                w_i, h_i, rotatable_i = rectangles[i]
                w_j, h_j, rotatable_j = rectangles[j]
                w_i_rot = w_i if not rotated[i] else h_i
                h_i_rot = h_i if not rotated[i] else w_i
                w_j_rot = w_j if not rotated[j] else h_j
                h_j_rot = h_j if not rotated[j] else w_j
                
                # Left of
                left_of = model.NewBoolVar(f'left_{i}_{j}')
                model.Add(x[i] + w_i_rot <= x[j]).OnlyEnforceIf(placed[i] & placed[j] & left_of)
                
                # Right of
                right_of = model.NewBoolVar(f'right_{i}_{j}')
                model.Add(x[j] + w_j_rot <= x[i]).OnlyEnforceIf(placed[i] & placed[j] & right_of)
                
                # Below
                below = model.NewBoolVar(f'below_{i}_{j}')
                model.Add(y[i] + h_i_rot <= y[j]).OnlyEnforceIf(placed[i] & placed[j] & below)
                
                # Above
                above = model.NewBoolVar(f'above_{i}_{j}')
                model.Add(y[j] + h_j_rot <= y[i]).OnlyEnforceIf(placed[i] & placed[j] & above)
                
                # At least one condition must hold
                model.Add(left_of + right_of + below + above >= 1).OnlyEnforceIf(placed[i] & placed[j])
        
        # Maximize number of placed rectangles
        model.Maximize(sum(placed))
        
        # Solve with time limit
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = kwargs.get('time_limit', 900.0)
        status = solver.Solve(model)
        
        # Extract solution
        placements = []
        for i in range(n):
            if solver.Value(placed[i]):
                x_val = solver.Value(x[i])
                y_val = solver.Value(y[i])
                rotated_val = solver.Value(rotated[i]) == 1
                placements.append((i, x_val, y_val, rotated_val))
        
        return placements
