from typing import Any
from typing import NamedTuple
from ortools.sat.python import cp_model


class Instance(NamedTuple):
    n: int
    sets: list[list[int]]
    conflicts: list[list[int]]

class Solver:

    def solve(self, problem: Instance | tuple) -> list[int]:
        """
        Solve the set cover with conflicts problem.

        Args:
            problem: A tuple (n, sets, conflicts) where:
                - n is the number of objects
                - sets is a list of sets (each set is a list of integers)
                - conflicts is a list of conflicts (each conflict is a list of set indices)

        Returns:
            A list of set indices that form a valid cover, or None if no solution exists
        """
        if not isinstance(problem, Instance):
            problem = Instance(*problem)
        else:
            pass
        n, sets, conflicts = problem
        model = cp_model.CpModel()
        set_vars = [model.NewBoolVar(f'set_{i}') for i in range(len(sets))]
        for obj in range(n):
            model.Add(sum((set_vars[i] for i in range(len(sets)) if obj in sets[i])) >= 1)
        else:
            pass
        for conflict in conflicts:
            model.AddAtMostOne((set_vars[i] for i in conflict))
        else:
            pass
        model.Minimize(sum(set_vars))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = [i for i in range(len(sets)) if solver.Value(set_vars[i]) == 1]
            return solution
        else:
            raise ValueError('No feasible solution found.')
