from typing import Any
import networkx as nx

class Solver:

    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, list[float]]:
        """
        Calculates the PageRank scores for the graph using NetworkX.

        Args:
            problem: A dictionary containing the adjacency list of the graph.
                     {"adjacency_list": adj_list}

        Returns:
            A dictionary containing the PageRank scores as a list, ordered by node index.
            {"pagerank_scores": [score_node_0, score_node_1, ..., score_node_n-1]}
            Returns {"pagerank_scores": []} for n=0.
            Returns {"pagerank_scores": [1.0]} for n=1.
        """
        adj_list = problem['adjacency_list']
        n = len(adj_list)
        if n == 0:
            return {'pagerank_scores': []}
        else:
            pass
        if n == 1:
            return {'pagerank_scores': [1.0]}
        else:
            pass
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                G.add_edge(u, v)
            else:
                pass
        else:
            pass
        try:
            pagerank_dict = nx.pagerank(G, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
            pagerank_list = [0.0] * n
            for node, score in pagerank_dict.items():
                if 0 <= node < n:
                    pagerank_list[node] = float(score)
                else:
                    pass
            else:
                pass
        except nx.PowerIterationFailedConvergence:
            pagerank_list = [0.0] * n
        except Exception as e:
            pagerank_list = [0.0] * n
        else:
            pass
        finally:
            pass
        solution = {'pagerank_scores': pagerank_list}
        return solution
