from typing import Any
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:

    def __init__(self):
        self.directed = False
        self.method = 'D'

    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solves the all-pairs shortest path problem using scipy.sparse.csgraph.shortest_path.
        :param problem: A dictionary representing the graph in CSR components.
        :return: A dictionary with key "distance_matrix":
                 "distance_matrix": The matrix of shortest path distances (list of lists).
                                     np.inf indicates no path.
        """
        try:
            graph_csr = scipy.sparse.csr_matrix((problem['data'], problem['indices'], problem['indptr']), shape=problem['shape'])
        except Exception as e:
            return {'distance_matrix': []}
        else:
            pass
        finally:
            pass
        try:
            dist_matrix = scipy.sparse.csgraph.shortest_path(csgraph=graph_csr, method=self.method, directed=self.directed)
        except Exception as e:
            return {'distance_matrix': []}
        else:
            pass
        finally:
            pass
        dist_matrix_list = [[None if np.isinf(d) else d for d in row] for row in dist_matrix]
        solution = {'distance_matrix': dist_matrix_list}
        return solution
