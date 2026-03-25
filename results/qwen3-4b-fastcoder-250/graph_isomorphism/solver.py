import networkx as nx
from typing import Dict, List, Any

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        n = problem["num_nodes"]
        edges_g1 = problem["edges_g1"]
        edges_g2 = problem["edges_g2"]
        
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_nodes_from(range(n))
        G2.add_nodes_from(range(n))
        G1.add_edges_from(edges_g1)
        G2.add_edges_from(edges_g2)
        
        gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        iso_map = next(gm.isomorphisms_iter())
        return {"mapping": [iso_map[i] for i in range(n)]}
