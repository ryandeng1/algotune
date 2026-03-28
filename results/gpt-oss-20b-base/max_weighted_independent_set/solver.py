from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, list]) -> list[int]:
        """
        Max-Weight Independent Set (MWIS) solver using OR-Tools CP-SAT.
        """
        adj_matrix = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(weights)
        model = cp_model.CpModel()

        # Decision variables: one boolean per vertex
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add edge constraints only for existing edges.
        # For a large matrix this avoids O(n²) loops when the graph is sparse.
        for i in range(n):
            row = adj_matrix[i]
            for j in range(i + 1, n):
                if row[j]:
                    model.Add(x[i] + x[j] <= 1)

        # Objective: maximize total weight of selected vertices
        model.Maximize(sum(weights[i] * x[i] for i in range(n)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # optional time limit
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(x[i])]
        return []