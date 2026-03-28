from ortools.sat.python import cp_model
from typing import NamedTuple, Iterable, List, Tuple, Union

class Instance(NamedTuple):
    n: int
    sets: List[List[int]]
    conflicts: List[List[int]]

class Solver:
    def solve(self, problem: Union[Instance, Tuple[int, List[List[int]], List[List[int]]]]) -> List[int]:
        if not isinstance(problem, Instance):
            problem = Instance(*problem)

        n, sets, conflicts = problem
        m = len(sets)

        # Pre-compute which sets cover each object
        covers: List[List[int]] = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for obj in s:
                covers[obj].append(i)

        model = cp_model.CpModel()
        set_vars = [model.NewBoolVar(f'set_{i}') for i in range(m)]

        # Coverage constraints
        for obj_covers in covers:
            if obj_covers:  # avoid empty
                model.Add(sum(set_vars[i] for i in obj_covers) >= 1)

        # Conflict constraints
        for conflict in conflicts:
            if len(conflict) > 1:
                model.AddAtMostOne(set_vars[i] for i in conflict)

        model.Minimize(sum(set_vars))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0
        solver.parameters.num_search_workers = 8

        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(m) if solver.Value(set_vars[i])]
        raise ValueError('No feasible solution found.')