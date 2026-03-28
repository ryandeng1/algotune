from typing import NamedTuple, Iterable
from ortools.sat.python import cp_model

class Instance(NamedTuple):
    n: int
    sets: list[list[int]]
    conflicts: list[list[int]]

class Solver:
    def solve(self, problem: Instance | tuple) -> list[int]:
        """Optimal set‑cover with conflicts using CpModel."""
        if not isinstance(problem, Instance):
            problem = Instance(*problem)
        n, sets, conflicts = problem

        # Pre‑compute which sets cover each object
        covers = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for obj in s:
                covers[obj].append(i)

        model = cp_model.CpModel()
        num_sets = len(sets)
        set_vars = [model.NewBoolVar(f'set_{i}') for i in range(num_sets)]

        # Coverage constraints
        for obj, idxs in enumerate(covers):
            if idxs:   # object is covered by at least one set
                model.Add(sum(set_vars[i] for i in idxs) >= 1)

        # Conflict constraints
        for conflict in conflicts:
            if conflict:
                model.AddAtMostOne(set_vars[i] for i in conflict)

        # Objective: minimise number of selected sets
        model.Minimize(sum(set_vars))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300  # optional
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(num_sets) if solver.Value(set_vars[i])]
        raise ValueError('No feasible solution found.')