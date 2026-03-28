from typing import Any
import networkx as nx


SolutionType = dict[str, int]

class Solver:

    def solve(self, problem: dict[str, Any]) -> SolutionType:
        try:
            n = problem.get('num_nodes', 0)
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(problem['edges'])
            cc = nx.number_connected_components(G)
            return {'number_connected_components': cc}
        except Exception as e:
            return {'number_connected_components': -1}
        else:
            pass
        finally:
            pass
