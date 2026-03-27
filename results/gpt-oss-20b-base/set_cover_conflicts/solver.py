from typing import NamedTuple, Tuple, List, Union
from ortools.sat.python import cp_model


class Instance(NamedTuple):
    n: int
    sets: List[List[int]]
    conflicts: List[List[int]]


class Solver:
    def solve(self, problem: Union[Instance, Tuple[int, List[List[int]], List[List[int]]]]) -> List[int]:
        """
        Solve the set cover with conflicts problem using OR-Tools CP-SAT.

        Args:
            problem: either an Instance or a tuple (n, sets, conflicts).

        Returns:
            List of selected set indices that form a valid cover.
        """
        if not isinstance(problem, Instance):
            problem = Instance(*problem)

        n, sets, conflicts = problem
        m = len(sets)

        model = cp_model.CpModel()

        # Binary variables for each set
        set_vars = [model.NewBoolVar(f'set_{i}') for i in range(m)]

        # Pre‑compute which sets cover each object
        obj_to_sets = [[] for _ in range(n)]
        for i, s in enumerate(sets):
            for obj in s:
                obj_to_sets[obj].append(i)

        # Coverage constraints: each object must be covered by at least one selected set
        for obj, set_indices in enumerate(obj_to_sets):
            if set_indices:  # avoid adding constraints for objects that appear in no set
                model.Add(sum(set_vars[i] for i in set_indices) >= 1)

        # Conflict constraints: at most one set from each conflict group can be selected
        for conflict in conflicts:
            if conflict:
                model.AddAtMostOne([set_vars[i] for i in conflict])

        # Objective: minimize total number of selected sets
        model.Minimize(sum(set_vars))

        # Solver options: use fast linear search with a relatively small time limit
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # adjust if needed

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i, var in enumerate(set_vars) if solver.Value(var) == 1]
        raise ValueError("No feasible solution found.")