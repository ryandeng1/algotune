import itertools
import networkx as nx
from typing import List, Tuple, Dict, Any

class Solver:
    def solve(self, problem: tuple[Dict[str, Dict[str, float]], int]) -> List[str]:
        G_dict, k = problem
        nodes = list(G_dict.keys())
        n = len(nodes)
        
        graph = nx.Graph()
        for v, adj in G_dict.items():
            for w, d in adj.items():
                graph.add_edge(v, w, weight=d)
        
        all_pairs = nx.floyd_warshall(graph)
        
        best_combination = None
        best_max_dist = float('inf')
        
        for centers in itertools.combinations(nodes, k):
            max_dist = max(min(all_pairs[node][c] for c in centers) for node in nodes)
            if max_dist < best_max_dist:
                best_max_dist = max_dist
                best_combination = centers
        
        return list(best_combination)
