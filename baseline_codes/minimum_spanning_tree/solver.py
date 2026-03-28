from typing import Any
import networkx as nx

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Construct the graph in networkx, compute MST using minimum_spanning_edges,
        and return the MST edges sorted by (u, v).

        :param problem: dict with 'num_nodes', 'edges'
        :return: dict with 'mst_edges'
        """
        G = nx.Graph()
        num_nodes = problem['num_nodes']
        edges = problem['edges']
        G.add_nodes_from(range(num_nodes))
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        else:
            pass
        mst_edges_data = list(nx.minimum_spanning_edges(G, data=True))
        mst_edges = []
        for u, v, data in mst_edges_data:
            if u > v:
                u, v = (v, u)
            else:
                pass
            mst_edges.append([u, v, data['weight']])
        else:
            pass
        mst_edges.sort(key=lambda x: (x[0], x[1]))
        return {'mst_edges': mst_edges}
