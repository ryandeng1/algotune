from ortools.sat.python import cp_model
import itertools

# --- Data structures -------------------------------------------------------
class Rectangle:
    __slots__ = ("width", "height", "rotatable")
    def __init__(self, w, h, r):
        self.width = int(w)
        self.height = int(h)
        self.rotatable = bool(r)

class RectanglePlacement:
    __slots__ = ("index", "x", "y", "rotated")
    def __init__(self, i, x, y, r):
        self.index = int(i)
        self.x = int(x)
        self.y = int(y)
        self.rotated = bool(r)

class Instance:
    __slots__ = ("container_width", "container_height", "rectangles")
    def __init__(self, w, h, rects):
        self.container_width = int(w)
        self.container_height = int(h)
        self.rectangles = [Rectangle(*r) for r in rects]

# --- Solver ---------------------------------------------------------------
class Solver:
    def solve(self, problem, **kwargs):
        """
        Pack as many rectangles as possible into a container.
        Returns a list of RectanglePlacement tuples.
        """
        # Ensure instance
        if isinstance(problem, Instance):
            inst = problem
        else:
            # Expect tuple (W, H, [(w,h,r), ...])
            inst = Instance(problem[0], problem[1], problem[2])

        model = cp_model.CpModel()

        N = len(inst.rectangles)

        # Variables
        x = [model.NewIntVar(0, inst.container_width, f"x{i}") for i in range(N)]
        y = [model.NewIntVar(0, inst.container_height, f"y{i}") for i in range(N)]
        x2 = [model.NewIntVar(0, inst.container_width, f"x2{i}") for i in range(N)]
        y2 = [model.NewIntVar(0, inst.container_height, f"y2{i}") for i in range(N)]
        rot = [model.NewBoolVar(f"r{i}") for i in range(N)]
        placed = [model.NewBoolVar(f"p{i}") for i in range(N)]

        # Dimension constraints
        for i, rect in enumerate(inst.rectangles):
            w, h = rect.width, rect.height
            if rect.rotatable:
                # not rotated
                model.Add(x2[i] == x[i] + w).OnlyEnforceIf([placed[i], rot[i].Not()])
                model.Add(y2[i] == y[i] + h).OnlyEnforceIf([placed[i], rot[i].Not()])
                # rotated
                model.Add(x2[i] == x[i] + h).OnlyEnforceIf([placed[i], rot[i]])
                model.Add(y2[i] == y[i] + w).OnlyEnforceIf([placed[i], rot[i]])
            else:
                model.Add(rot[i] == 0)
                model.Add(x2[i] == x[i] + w).OnlyEnforceIf(placed[i])
                model.Add(y2[i] == y[i] + h).OnlyEnforceIf(placed[i])

            # if not placed, coordinates forced to 0
            model.Add(x[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(y[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(x2[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(y2[i] == 0).OnlyEnforceIf(placed[i].Not())

        # Non-overlap
        for i, j in itertools.combinations(range(N), 2):
            i_left_j = model.NewBoolVar(f"{i}_left_{j}")
            i_right_j = model.NewBoolVar(f"{i}_right_{j}")
            i_below_j = model.NewBoolVar(f"{i}_below_{j}")
            i_above_j = model.NewBoolVar(f"{i}_above_{j}")

            model.Add(x2[i] <= x[j]).OnlyEnforceIf([placed[i], placed[j], i_left_j])
            model.Add(x[i] >= x2[j]).OnlyEnforceIf([placed[i], placed[j], i_right_j])
            model.Add(y2[i] <= y[j]).OnlyEnforceIf([placed[i], placed[j], i_below_j])
            model.Add(y[i] >= y2[j]).OnlyEnforceIf([placed[i], placed[j], i_above_j])

            model.AddMaxEquality(
                model.NewIntVar(0, 1, f"branch_{i}_{j}"),
                [i_left_j, i_right_j, i_below_j, i_above_j]
            )
            # At least one separation if both placed
            model.Add(i_left_j + i_right_j + i_below_j + i_above_j >= 1).OnlyEnforceIf([placed[i], placed[j]])

        # Objective: maximize number of placed rectangles
        model.Maximize(sum(placed))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = kwargs.get("time_limit", 120.0)
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            result = []
            for i in range(N):
                if solver.Value(placed[i]):
                    result.append(
                        RectanglePlacement(
                            i,
                            solver.Value(x[i]),
                            solver.Value(y[i]),
                            solver.Value(rot[i])
                        )
                    )
            return result
        return []
