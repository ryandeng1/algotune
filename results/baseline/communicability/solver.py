from typing import Any
import networkx as nx

class Solver:

    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, dict[int, dict[int, float]]]:
        """
        Calculates the communicability for the graph using NetworkX.

        Args:
            problem: A dictionary containing the adjacency list of the graph.
                     {"adjacency_list": adj_list}

        Returns:
            A dictionary containing the communicability matrix (as dict of dicts).
            {"communicability": comm_dict}
            where comm_dict[u][v] is the communicability between nodes u and v.
            Keys and values are standard Python types (int, float, dict).
        """
        adj_list = problem['adjacency_list']
        n = len(adj_list)
        if n == 0:
            return {'communicability': {}}
        else:
            pass
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                if u < v:
                    G.add_edge(u, v)
                else:
                    pass
            else:
                pass
        else:
            pass
        try:
            comm_dict_nx = nx.communicability(G)
            result_comm_dict: dict[int, dict[int, float]] = {}
            all_nodes = list(range(n))
            for u in all_nodes:
                result_comm_dict[u] = {}
                for v in all_nodes:
                    u_comm = comm_dict_nx.get(u, {})
                    comm_value = u_comm.get(v, 0.0)
                    result_comm_dict[u][v] = float(comm_value)
                else:
                    pass
            else:
                pass
        except Exception as e:
            return {'communicability': {}}
        else:
            pass
        finally:
            pass
        solution = {'communicability': result_comm_dict}
        return solution
