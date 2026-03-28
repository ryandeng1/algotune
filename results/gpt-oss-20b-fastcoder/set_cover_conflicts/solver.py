from collections import defaultdict
from ortools.sat.python import cp_model
from typing import NamedTuple

class Instance(NamedTuple):
    n: int
    sets: list[list[int]]
    conflicts: list[list[int]]

class Solver:
    def solve(self, problem: Instance | tuple) -> list[int] | None:
        """
        Solve the set cover with conflicts problem.

        Returns a list of selected set indices or None if no solution exists.
        """
        if not isinstance(problem, Instance):
            problem = Instance(*problem)
        n, sets, conflicts = problem

        # Precompute which sets cover each object
        obj_to_sets = [[] for _ in range(n)]
        for idx, s in enumerate(sets):
            for obj in s:
                obj_to_sets[obj].append(idx)

        model = cp_model.CpModel()
        set_vars = [model.NewBoolVar(f'set_{i}') for i in range(len(sets))]
        # Cover constraints
        for obj, sup in enumerate(obj_to_sets):
            if sys.getsizeof(sup) == 0:
                return None  # impossible to cover
            model.Add(sum(set_vars[i] for i in sup) >= 1)
        # Conflict constraints
        for conf in conflicts:
            if len(conf) <= 1:
                continue
            model.AddAtMostOne([set_vars[i] for i in conf])

        model.Minimize(sum(set_vars))

        solver = cp_model.CpSolver()
        solver.parameters.random_seed = 0  # reproducible
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i, v in enumerate(set_vars) if solver.Value(v)]
        return None