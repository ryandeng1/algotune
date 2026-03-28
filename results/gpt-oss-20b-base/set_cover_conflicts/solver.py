from typing import NamedTuple
from ortools.sat.python import cp_model


class Instance(NamedTuple):
    n: int
    sets: list[list[int]]
    conflicts: list[list[int]]


class Solver:
    def solve(self, problem: Instance | tuple) -> list[int]:
        if not isinstance(problem, Instance):
            problem = Instance(*problem)

        n, sets, conflicts = problem
        model = cp_model.CpModel()

        # Decision variables: whether a set is chosen
        set_vars = [model.NewBoolVar(f'set_{i}') for i in range(len(sets))]

        # Pre‑compute for each object the list of sets that contain it
        covers = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for o in s:
                covers[o].append(set_vars[i])

        # Cover constraints
        for obj_sets in covers:
            if obj_sets:  # object is covered by at least one set
                model.Add(sum(obj_sets) >= 1)

        # Conflict constraints – at most one set in each conflict
        for conflict in conflicts:
            model.AddAtMostOne([set_vars[i] for i in conflict])

        # Objective: minimize the number of selected sets
        model.Minimize(sum(set_vars))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i, var in enumerate(set_vars) if solver.Value(var)]
        raise ValueError("No feasible solution found.")