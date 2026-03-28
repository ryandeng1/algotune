from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, list]) -> list[int]:
        """
        Solves the MWIS problem using CP-SAT.

        Parameters
        ----------
        problem : dict
            Must contain keys:
                * 'adj_matrix' – square adjacency matrix (list of lists of bool/int)
                * 'weights'    – list of weights for each vertex

        Returns
        -------
        list[int]
            Indices of vertices selected in a maximum weight independent set.
        """
        adj = problem["adj_matrix"]
        weights = problem["weights"]
        n = len(adj)

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(n)]

        # Add edge constraints once for each undirected pair (i, j) with adjacency == 1
        for i in range(n):
            row = adj[i]
            for j in range(i + 1, n):
                if row[j]:
                    model.Add(x[i] + x[j] <= 1)

        # Objective: maximize sum(weights[i] * x[i])
        model.Maximize(sum(int(ws) * x[i] for i, ws in enumerate(weights)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0  # allow up to 10 s, can be tuned
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(x[i])]
        return []