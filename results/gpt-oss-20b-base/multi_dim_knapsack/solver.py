import typing
from ortools.sat.python import cp_model

class Solver:
    def solve(
        self,
        problem: typing.Union[typing.Tuple, list, dict, 'MultiDimKnapsackInstance']
    ) -> typing.List[int]:
        """
        Solve a Multi-Dimensional Knapsack Problem.

        Parameters
        ----------
        problem
            Either:
            * a tuple/list ``(value, demand, supply)``
            * a dict with keys 'value', 'demand', 'supply'
            * an object with attributes `value`, `demand`, `supply`.

        Returns
        -------
        list[int]
            List of selected item indices that maximize total value
            while respecting all resource constraints.
        """
        # unpack problem in a flexible manner
        if isinstance(problem, tuple) or isinstance(problem, list):
            if len(problem) != 3:
                return []
            value, demand, supply = problem
        elif isinstance(problem, dict):
            try:
                value = problem["value"]
                demand = problem["demand"]
                supply = problem["supply"]
            except KeyError:
                return []
        else:
            # assume object has required attributes
            try:
                value = problem.value
                demand = problem.demand
                supply = problem.supply
            except Exception:
                return []

        n = len(value)
        k = len(supply)

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # resource constraints
        for r in range(k):
            model.Add(
                sum(x[i] * demand[i][r] for i in range(n)) <= supply[r]
            )
        # objective
        model.Maximize(sum(x[i] * value[i] for i in range(n)))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i])]
        return []
