from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solve a minimum dominating set problem using OR-Tools CP-SAT.
        """
        n = len(problem)

        # Build adjacency lists once
        adj = [set(i for i, v in enumerate(row) if v) for row in problem]

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Each node or one of its neighbours must be chosen
        for i in range(n):
            # Sum of the node itself and its neighbours
            neigh = [x[i]] + [x[j] for j in adj[i]]
            model.Add(sum(neigh) >= 1)

        # Minimise the total number of selected nodes
        model.Minimize(sum(x))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0   # optional bound

        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i, var in enumerate(x) if solver.Value(var)]
        return []