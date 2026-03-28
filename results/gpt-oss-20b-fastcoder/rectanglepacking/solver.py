from typing import NamedTuple, List

class Rectangle(NamedTuple):
    width: int
    height: int
    rotatable: bool

class Instance(NamedTuple):
    container_width: int
    container_height: int
    rectangles: List[Rectangle]

class RectanglePlacement(NamedTuple):
    i: int
    x: int
    y: int
    rotated: bool

class Solver:
    def solve(self, problem: Instance) -> List[RectanglePlacement]:
        from ortools.sat.python import cp_model

        rects = problem.rectangles
        n = len(rects)
        model = cp_model.CpModel()

        # Variables
        x1 = [model.NewIntVar(0, problem.container_width, f'x1_{i}') for i in range(n)]
        y1 = [model.NewIntVar(0, problem.container_height, f'y1_{i}') for i in range(n)]
        x2 = [model.NewIntVar(0, problem.container_width, f'x2_{i}') for i in range(n)]
        y2 = [model.NewIntVar(0, problem.container_height, f'y2_{i}') for i in range(n)]
        rot = [model.NewBoolVar(f'rot_{i}') for i in range(n)]
        placed = [model.NewBoolVar(f'p_{i}') for i in range(n)]

        # Size constraints
        for i, r in enumerate(rects):
            if r.rotatable:
                # non‑rotated
                model.Add(x2[i] == x1[i] + r.width).OnlyEnforceIf([placed[i], rot[i].Not()])
                model.Add(y2[i] == y1[i] + r.height).OnlyEnforceIf([placed[i], rot[i].Not()])
                # rotated
                model.Add(x2[i] == x1[i] + r.height).OnlyEnforceIf([placed[i], rot[i]])
                model.Add(y2[i] == y1[i] + r.width).OnlyEnforceIf([placed[i], rot[i]])
            else:
                model.Add(x2[i] == x1[i] + r.width).OnlyEnforceIf(placed[i])
                model.Add(y2[i] == y1[i] + r.height).OnlyEnforceIf(placed[i])
                model.Add(rot[i] == 0)
            # Unplaced rectangles are at origin
            model.Add(x1[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(y1[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(x2[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(y2[i] == 0).OnlyEnforceIf(placed[i].Not())

        # Non‑overlap
        for i in range(n):
            for j in range(i + 1, n):
                if_ok = [
                    placed[i],
                    placed[j],
                ]
                left = model.NewBoolVar(f'{i}_L_{j}')
                model.Add(x2[i] <= x1[j]).OnlyEnforceIf([*if_ok, left])
                right = model.NewBoolVar(f'{i}_R_{j}')
                model.Add(x1[i] >= x2[j]).OnlyEnforceIf([*if_ok, right])
                below = model.NewBoolVar(f'{i}_B_{j}')
                model.Add(y2[i] <= y1[j]).OnlyEnforceIf([*if_ok, below])
                above = model.NewBoolVar(f'{i}_A_{j}')
                model.Add(y1[i] >= y2[j]).OnlyEnforceIf([*if_ok, above])
                model.Add(left + right + below + above >= 1).OnlyEnforceIf(if_ok)

        # Objective
        model.Maximize(sum(placed))

        # Solver
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 900.0
        solver.parameters.log_search_progress = True
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [
                RectanglePlacement(i, solver.Value(x1[i]), solver.Value(y1[i]), bool(solver.Value(rot[i])))
                for i in range(n)
                if solver.Value(placed[i])
            ]
        return []

    def _typesafe_instance(self, instance) -> Instance:
        if isinstance(instance, Instance):
            return instance
        return Instance(instance[0], instance[1], [Rectangle(*r) for r in instance[2]])