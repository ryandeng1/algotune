# solver.py
"""
A highly optimised CP‑SAT solver for the Vehicle Routing Problem.
"""

from ortools.sat.python import cp_model
from typing import Any, List, Dict

class Solver:
    """
    Solver based on Google OR‑Tools CP‑SAT.
    """
    def __init__(self) -> None:
        # Pre‑allocate a reusable constant list for all zero rows
        self._zero_row: List[int] = []

    @staticmethod
    def _build_solver(
        D: List[List[int]], K: int, depot: int
    ) -> cp_model.Solver:
        n = len(D)
        model = cp_model.CpModel()

        # Decision variables: x[i][j] == 1 if the route goes from i to j
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

        # Degree constraints
        for i in range(n):
            if i == depot:
                continue
            model.Add(sum(x[j, i] for j in range(n) if j != i) == 1)
            model.Add(sum(x[i, j] for j in range(n) if j != i) == 1)

        # Exactly K vehicles leave/arrive at the depot
        model.Add(sum(x[depot, j] for j in range(n) if j != depot) == K)
        model.Add(sum(x[i, depot] for i in range(n) if i != depot) == K)

        # Sub‑tour elimination (Miller–Rust–Seymour)
        # u[i] : position of node i in a vehicle’s tour (1 … n-1)
        u: Dict[int, cp_model.IntVar] = {}
        for i in range(n):
            if i == depot:
                continue
            u[i] = model.NewIntVar(1, n - 1, f"u_{i}")

        for i in range(n):
            if i == depot:
                continue
            for j in range(n):
                if j == depot or j == i:
                    continue
                # u[i] + 1 <= u[j] + (n-1)*(1-x[i][j])
                model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[i, j]))

        # Objective: minimise total distance travelled
        obj = sum(D[i][j] * x[i, j] for i, j in x)
        model.Minimize(obj)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0  # optional speed‑up
        return solver, model, x, n, depot

    @staticmethod
    def _extract_routes(
        solver: cp_model.CpSolver,
        x: Dict[tuple[int, int], cp_model.IntVar],
        n: int,
        depot: int,
    ) -> List[List[int]]:
        routes: List[List[int]] = []

        # For each vehicle start node j (the node directly after the depot)
        for j in range(n):
            if j == depot:
                continue
            if solver.Value(x[depot, j]) == 1:
                route = [depot, j]
                current = j
                while current != depot:
                    for k in range(n):
                        if current != k and solver.Value(x[current, k]) == 1:
                            route.append(k)
                            current = k
                            break
                routes.append(route)

        return routes

    def solve(self, problem: Dict[str, Any]) -> List[List[int]]:
        """
        Solve a vehicle routing problem.

        Parameters
        ----------
        problem : dict
            `{ "D": List[List[int]],
               "K": int,
               "depot": int }`

        Returns
        -------
        List[List[int]]
            One list per vehicle: the route starting and ending at the depot.
            An empty list if the problem is infeasible.
        """
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        solver, model, x, n, depot = self._build_solver(D, K, depot)

        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._extract_routes(solver, x, n, depot)
        return []