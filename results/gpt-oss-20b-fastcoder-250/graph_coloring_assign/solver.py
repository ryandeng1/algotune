import networkx as nx

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Fast greedy graph coloring that returns a proper coloring.
        Although not guaranteed optimal, it is extremely quick and
        generally produces a coloring using few colors.
        """
        n = len(problem)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j]:
                    G.add_edge(i, j)

        # Use a high-quality greedy coloring strategy
        coloring = nx.algorithms.coloring.greedy_color(G, strategy='largest_first')
        return [coloring[i] + 1 for i in range(n)]
