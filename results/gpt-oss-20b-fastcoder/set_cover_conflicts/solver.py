from typing import Any, NamedTuple, List, Union

from ortools.sat.python import cp_model


class Instance(NamedTuple):
    n: int
    sets: List[List[int]]
    conflicts: List[List[int]]


class Solver:
    """Fast solver for set‑cover with conflicts using OR‑Tools CP‑SAT."""

    def solve(self, problem: Union[Instance, tuple]) -> List[int]:
        if not isinstance(problem, Instance):
            problem = Instance(*problem)

        n, sets, conflicts = problem
        num_sets = len(sets)

        model = cp_model.CpModel()
        set_vars = [model.NewBoolVar(f'set_{i}') for i in range(num_sets)]

        # Pre‑compute which sets cover each object
        obj_to_sets = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for v in s:
                obj_to_sets[v].append(set_vars[i])

        # Object coverage constraints
        for covers in obj_to_sets:
            model.Add(sum(covers) >= 1)

        # Conflict constraints
        for clash in conflicts:
            model.AddAtMostOne([set_vars[i] for i in clash])

        model.Minimize(sum(set_vars))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60  # optional timeout
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i, v in enumerate(set_vars) if solver.Value(v) == 1]
        raise ValueError("No feasible solution found.")