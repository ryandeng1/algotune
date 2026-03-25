# solver.py
import pulp
from typing import List, Any

class Solver:
    def solve(self, problem: List[List[int]]) -> List[int]:
        """
        Solves the minimum vertex cover problem by formulating it as a
        binary integer program and solving with the default CBC solver.
        The objective is to minimize the number of vertices selected
        subject to covering all edges.
        """
        n = len(problem)
        if n == 0:
            return []

        # Create binary variables: 1 if vertex i is in the cover
        x = [pulp.LpVariable(f"x{i}", 0, 1, pulp.LpBinary) for i in range(n)]

        # Define the problem: minimize sum of x[i]
        prob = pulp.LpProblem("VertexCover", pulp.LpMinimize)
        prob += pulp.lpSum(x)

        # Add cover constraints: for each edge (i, j), x[i] + x[j] >= 1
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j]:
                    prob += x[i] + x[j] >= 1

        # Solve using CBC (default solver in pulp)
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Extract solution
        cover = [i for i in range(n) if pulp.value(x[i]) > 0.5]
        return cover
