from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solve the maximum clique problem using ortools CP‑SAT.

        :param problem: 2‑D adjacency matrix (list of lists) of the graph.
        :return: Indices of nodes that form a maximum clique (empty list if no solution).
        """
        n = len(problem)

        model = cp_model.CpModel()
        # Boolean vars: x[i] == 1 if node i is in the clique
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Add constraints: for any non‑edge (i, j) at most one of the nodes can be chosen
        for i in range(n):
            row_i = problem[i]
            for j in range(i + 1, n):
                if row_i[j] == 0:          # no edge between i and j
                    model.Add(x[i] + x[j] <= 1)

        # Objective: maximize the size of the clique
        model.Maximize(sum(x))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0          # optional timeout (adjust as needed)
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [i for i in range(n) if solver.Value(x[i]) == 1]
        return []