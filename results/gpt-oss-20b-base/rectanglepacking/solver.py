from __future__ import annotations

from typing import NamedTuple, List

# -----------------------------------------------------------------------
# Data structures – small @dataclass‑like NamedTuples for speed and clarity
# -----------------------------------------------------------------------

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


# -----------------------------------------------------------------------
# Solver – compact, fast and self‑contained
# -----------------------------------------------------------------------

class Solver:

    # -------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------
    def solve(self, problem: Instance) -> List[RectanglePlacement]:
        """
        Solve the rectangle packing problem using OR‑Tools CP‑SAT.

        The model places as many rectangles as possible into the container,
        allowing optional rotations.  The solver is limited by a hard time
        constraint (default 900 s), which can be tuned by the caller
        (currently unused but kept for compatibility).

        Returns a list of placements – one entry for each rectangle that
        was actually positioned.
        """
        problem = self._typesafe_instance(problem)

        # All heavy work is delegated to a small helper class.
        model = _RectangleKnapsackModel(problem)
        return model.solve(900.0)

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------
    @staticmethod
    def _typesafe_instance(instance) -> Instance:
        """
        Convert user‑supplied data (tuple, list, ...) into the
        canonical Instance type.
        """
        if isinstance(instance, Instance):
            return instance

        width, height, rects = instance
        return Instance(
            width,
            height,
            [Rectangle(*r) for r in rects],
        )


# -----------------------------------------------------------------------
# Helper class – implements the CP‑SAT model
# -----------------------------------------------------------------------
class _RectangleKnapsackModel:
    """
    Lightweight wrapper around OR‑Tools CP‑SAT.

    The class purposefully does *not* expose the full CP‑SAT API; it is
    a small, highly optimised sub‑module that can be instantiated multiple
    times for large benchmark loops without pulling in heavy packages each
    time.
    """

    # Import once – isolates the heavy dependency inside the module.
    from ortools.sat.python import cp_model

    def __init__(self, instance: Instance):
        self.instance = instance
        model = self.cp_model.CpModel()

        # Pre‑allocate arrays of variables – the CP‑SAT API is very cheap
        # when using list comprehension and flat indexing.
        n = len(instance.rectangles)
        cx = [model.NewIntVar(0, instance.container_width, f"x1_{i}") for i in range(n)]
        cy = [model.NewIntVar(0, instance.container_height, f"y1_{i}") for i in range(n)]

        rx = [model.NewIntVar(0, instance.container_width, f"x2_{i}") for i in range(n)]
        ry = [model.NewIntVar(0, instance.container_height, f"y2_{i}") for i in range(n)]

        rotated = [model.NewBoolVar(f"rot_{i}") for i in range(n)]
        placed = [model.NewBoolVar(f"plc_{i}") for i in range(n)]

        self._vars = (cx, cy, rx, ry, rotated, placed)

        w, h = instance.container_width, instance.container_height
        rects = instance.rectangles

        # Constraints for each rectangle – the triplets-of‑also‑enforce
        # pattern is replaced with simple implications using the CP‑SAT
        # add*If + add*IfNot helper methods which are slightly faster.
        for i, r in enumerate(rects):
            # Size
            if r.rotatable:
                model.Add(rx[i] == cx[i] + r.width).OnlyEnforceIf([placed[i], rotated[i].Not()])
                model.Add(ry[i] == cy[i] + r.height).OnlyEnforceIf([placed[i], rotated[i].Not()])

                model.Add(rx[i] == cx[i] + r.height).OnlyEnforceIf([placed[i], rotated[i]])
                model.Add(ry[i] == cy[i] + r.width).OnlyEnforceIf([placed[i], rotated[i]])
            else:
                model.Add(rx[i] == cx[i] + r.width).OnlyEnforceIf(placed[i])
                model.Add(ry[i] == cy[i] + r.height).OnlyEnforceIf(placed[i])
                model.Add(rotated[i] == 0)

            # Unplaced rectangles must be positioned at the origin
            model.Add(cx[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(cy[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(rx[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(ry[i] == 0).OnlyEnforceIf(placed[i].Not())

        # Non‑overlap: for every pair we define four boolean variables that
        # encode the relative positions.  The absolute constraint is the sum
        # of these booleans >= 1, which is more compact and quicker for CP‑SAT.
        for i in range(n):
            for j in range(i + 1, n):
                b_left = model.NewBoolVar(f"l_{i}_{j}")
                b_right = model.NewBoolVar(f"r_{i}_{j}")
                b_below = model.NewBoolVar(f"b_{i}_{j}")
                b_above = model.NewBoolVar(f"a_{i}_{j}")

                model.Add(rx[i] <= cx[j]).OnlyEnforceIf([placed[i], placed[j], b_left])
                model.Add(cx[i] >= rx[j]).OnlyEnforceIf([placed[i], placed[j], b_right])
                model.Add(ry[i] <= cy[j]).OnlyEnforceIf([placed[i], placed[j], b_below])
                model.Add(cy[i] >= ry[j]).OnlyEnforceIf([placed[i], placed[j], b_above])

                model.AddBoolOr([b_left, b_right, b_below, b_above]).OnlyEnforceIf([placed[i], placed[j]])

        # Objective – maximize the number of placed rectangles
        model.Maximize(sum(placed))

        self.model = model
        self._solver = self.cp_model.CpSolver()
        self._solver.parameters.max_time_in_seconds = 900.0
        # Disable search progress to reduce overhead
        self._solver.parameters.log_search_progress = False

    # -------------------------------------------------------------------
    # Solve the model and parse the result
    # -------------------------------------------------------------------
    def solve(self, time_limit: float) -> List[RectanglePlacement]:
        """
        Solve with a custom time limit in seconds.
        Returns the optimal placement set.
        """
        self._solver.parameters.max_time_in_seconds = time_limit
        status = self._solver.Solve(self.model)

        # CP‑SAT returns 0 for not found, 3 =FEASIBLE and 4 =OPTIMAL
        good = status in {self.cp_model.SolverStatus.OPTIMAL,
                          self.cp_model.SolverStatus.FEASIBLE}
        if not good:
            return []

        # Extract solution
        cx, cy, rx, ry, rot, plc = self._vars
        res = []
        for i, (x1, y1, _, _, r, p) in enumerate(zip(cx, cy, rx, ry, rot, plc)):
            if self._solver.Value(p):
                res.append(RectanglePlacement(i,
                                              int(self._solver.Value(x1)),
                                              int(self._solver.Value(y1)),
                                              bool(self._solver.Value(r))))
        return res