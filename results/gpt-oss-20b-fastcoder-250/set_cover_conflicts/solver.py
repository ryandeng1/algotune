from typing import NamedTuple, List, Union
from ortools.sat.python import cp_model


class Instance(NamedTuple):
    n: int
    sets: List[List[int]]
    conflicts: List[List[int]]


class Solver:
    def solve(self, problem: Union[Instance, tuple]) -> List[int]:
        """
        Solve the set cover with conflicts problem.

        Returns a list of set indices that form a valid cover.
        """
        if not isinstance(problem, Instance):
            problem = Instance(*problem)

        n, sets, conflicts = problem
        num_sets = len(sets)

        # Build a list of indices for each object for faster construction
        obj_to_sets: List[List[int]] = [[] for _ in range(n)]
        for s_idx, s in enumerate(sets):
            for obj in s:
                if 0 <= obj < n:
                    obj_to_sets[obj].append(s_idx)

        model = cp_model.CpModel()
        set_vars = [model.NewBoolVar(f"set_{i}") for i in range(num_sets)]

        # Cover constraints
        for obj, s_indices in enumerate(obj_to_sets):
            if s_indices:  # ensure there is at least one set covering the object
                model.Add(sum(set_vars[i] for i in s_indices) >= 1)
            else:
                # Object not in any set – unsatisfiable problem
                raise ValueError(f"Object {obj} cannot be covered by any set.")

        # Conflict constraints
        for conflict in conflicts:
            if conflict:
                model.AddAtMostOne(set_vars[i] for i in conflict)

        # Objective: minimize number of selected sets
        model.Minimize(sum(set_vars))

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(num_sets) if solver.Value(set_vars[i])]
        raise ValueError("No feasible solution found.")