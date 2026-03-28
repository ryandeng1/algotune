from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solve the maximum clique problem using OR‑Tools CP‑SAT.
        """
        n = len(problem)
        if n == 0:
            return []

        # Quick exit if the graph is empty
        if all(all(v == 0 for v in row) for row in problem):
            return []

        model = cp_model.CpModel()

        # Boolean variables for node selection
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add pairwise constraints for non‑adjacent vertices
        # Only iterate over upper triangle to avoid duplicates
        for i in range(n):
            row_i = problem[i]
            for j in range(i + 1, n):
                if row_i[j] == 0:   # not connected -> cannot both be in the clique
                    model.Add(nodes[i] + nodes[j] <= 1)

        # Objective: maximize number of chosen nodes
        model.Maximize(sum(nodes))

        # Solve with default settings (fast and memory efficient)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0  # optional limit
        solver.parameters.num_search_workers = 4     # use all cores

        status = solver.Solve(model)

        # Extract solution if found
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i, var in enumerate(nodes) if solver.Value(var)]
        return []