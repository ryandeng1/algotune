from typing import Any
import networkx as nx


class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, list[float]]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)

        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from((u, v) for u, neighbors in enumerate(adj_list) for v in neighbors)

        try:
            pagerank_dict = nx.pagerank(G, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
            pagerank_list = [float(pagerank_dict.get(i, 0.0)) for i in range(n)]
        except nx.PowerIterationFailedConvergence:
            pagerank_list = [0.0] * n
        except Exception as e:
            pagerank_list = [0.0] * n

        return {"pagerank_scores": pagerank_list}