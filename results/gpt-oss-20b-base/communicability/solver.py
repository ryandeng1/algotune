import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, dict[int, dict[int, float]]]:
        """
        Calculates the communicability matrix of an undirected graph.
        The communicability between nodes u and v is defined as the (u,v) entry of e^A,
        where A is the adjacency matrix of the graph.

        Args:
            problem: A dictionary containing the adjacency list of the graph.
                     {"adjacency_list": adj_list}

        Returns:
            A dictionary containing the communicability matrix (as dict of dicts).
            {"communicability": comm_dict}
            where comm_dict[u][v] is the communicability between nodes u and v.
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}

        # Build adjacency matrix efficiently
        A = np.zeros((n, n), dtype=float)
        # Since the graph is undirected, we only fill the upper triangle and mirror it
        for i, neighbors in enumerate(adj_list):
            for j in neighbors:
                if i < j:
                    A[i, j] = 1.0
                    A[j, i] = 1.0

        # Compute communicability matrix as exp(A)
        comm_matrix = expm(A)

        # Convert numpy array to nested dicts of Python floats
        comm_dict = {
            i: {j: float(comm_matrix[i, j]) for j in range(n)}
            for i in range(n)
        }

        return {"communicability": comm_dict}