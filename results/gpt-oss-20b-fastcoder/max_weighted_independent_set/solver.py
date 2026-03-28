from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, list]) -> list[int]:
        """
        Solves the Maximum Weighted Independent Set (MWIS) problem using OR-Tools CP-SAT.

        :param problem: dict with keys 'adj_matrix' (list of list of bool) and 'weights' (list of int)
        :return: list of indices of selected nodes
        """
        adj_matrix = problem['adj_matrix']
        weights = problem['weights']
        n = len(adj_matrix)

        model = cp_model.CpModel()

        # Decision variables
        x = [model.NewBoolVar(f'x_{i}') for i in range(n)]

        # Conflict constraints: for every edge (i, j) add x[i] + x[j] <= 1
        for i in range(n):
            row = adj_matrix[i]
            for j in range(i + 1, n):
                if row[j]:
                    model.Add(x[i] + x[j] <= 1)

        # Objective: maximize the sum of weights
        model.Maximize(sum(weights[i] * x[i] for i in range(n)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0  # or other timeout if needed
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(x[i])]
        return []