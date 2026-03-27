from __future__ import annotations
from typing import NamedTuple
import itertools
from ortools.sat.python import cp_model


class Rectangle(NamedTuple):
    width: int
    height: int
    rotatable: bool


class Instance(NamedTuple):
    container_width: int
    container_height: int
    rectangles: list[Rectangle]


class RectanglePlacement(NamedTuple):
    i: int
    x: int
    y: int
    rotated: bool


class Solver:
    def solve(self, problem: Instance) -> list[RectanglePlacement]:
        """Solve a packing problem using CP‑SAT with a compact formulation."""
        model = cp_model.CpModel()
        n = len(problem.rectangles)

        # coordinate variables - use signed ints, then clip with constraints
        x1 = [model.NewIntVar(0, problem.container_width, f"x1_{i}") for i in range(n)]
        y1 = [model.NewIntVar(0, problem.container_height, f"y1_{i}") for i in range(n)]
        x2 = [model.NewIntVar(0, problem.container_width, f"x2_{i}") for i in range(n)]
        y2 = [model.NewIntVar(0, problem.container_height, f"y2_{i}") for i in range(n)]

        rot = [model.NewBoolVar(f"rot_{i}") for i in range(n)]
        placed = [model.NewBoolVar(f"p_{i}") for i in range(n)]

        # Size constraints
        for i, rect in enumerate(problem.rectangles):
            w, h = rect.width, rect.height
            if rect.rotatable:
                model.Add(x2[i] == x1[i] + w).OnlyEnforceIf([placed[i], rot[i].Not()])
                model.Add(y2[i] == y1[i] + h).OnlyEnforceIf([placed[i], rot[i].Not()])

                model.Add(x2[i] == x1[i] + h).OnlyEnforceIf([placed[i], rot[i]])
                model.Add(y2[i] == y1[i] + w).OnlyEnforceIf([placed[i], rot[i]])
            else:
                model.Add(x2[i] == x1[i] + w).OnlyEnforceIf(placed[i])
                model.Add(y2[i] == y1[i] + h).OnlyEnforceIf(placed[i])
                model.Add(rot[i] == 0)

            # When not placed, all coords = 0
            model.Add(x1[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(y1[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(x2[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(y2[i] == 0).OnlyEnforceIf(placed[i].Not())

        # Non‑overlap constraints
        for i, j in itertools.combinations(range(n), 2):
            # If *both* rects are placed, enforce at least one separation
            b_left = model.NewBoolVar(f"{i}_L_{j}")
            b_right = model.NewBoolVar(f"{i}_R_{j}")
            b_below = model.NewBoolVar(f"{i}_B_{j}")
            b_above = model.NewBoolVar(f"{i}_A_{j}")

            model.Add(x2[i] <= x1[j]).OnlyEnforceIf([placed[i], placed[j], b_left])
            model.Add(x1[i] >= x2[j]).OnlyEnforceIf([placed[i], placed[j], b_right])
            model.Add(y2[i] <= y1[j]).OnlyEnforceIf([placed[i], placed[j], b_below])
            model.Add(y1[i] >= y2[j]).OnlyEnforceIf([placed[i], placed[j], b_above])

            model.Add(
                b_left + b_right + b_below + b_above >= 1
            ).OnlyEnforceIf([placed[i], placed[j]])

        # Objective: maximize the number of placed rectangles
        model.Maximize(sum(placed))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 900.0
        solver.parameters.log_search_progress = True
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [
                RectanglePlacement(
                    i=i,
                    x=solver.Value(x1[i]),
                    y=solver.Value(y1[i]),
                    rotated=solver.Value(rot[i]) == 1,
                )
                for i in range(n)
                if solver.Value(placed[i])
            ]
        return []