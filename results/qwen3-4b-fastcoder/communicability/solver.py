from typing import Any
import networkx as nx


class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, dict[int, dict[int, float]]]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        if n == 0:
            return {"communicability": {}}
        
        edges = []
        for u in range(n):
            for v in adj_list[u]:
                if u < v:
                    edges.append((u, v))
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        
        try:
            comm_dict_nx = nx.communicability(G)
            result_comm_dict = {
                u: {v: float(comm_dict_nx.get(u, {}).get(v, 0.0)) for v in range(n)}
                for u in range(n)
            }
        except Exception:
            return {"communicability": {}}
        
        return {"communicability": result_comm_dict}