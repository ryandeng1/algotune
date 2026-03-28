from typing import Any
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:

    def __init__(self):
        self.directed = False
        self.min_only = True

    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solves the shortest path problem from specified indices using scipy.sparse.csgraph.dijkstra.

        Returns only the distances.

        :param problem: A dictionary representing the graph (CSR) and source indices.
        :return: A dictionary with key "distances":
                 "distances": A list of shortest path distances from the source nodes.
                              If multiple sources, shape is (num_sources, n). If one source, shape is (n,).
                              Contains floats, uses np.inf for no path.
                              Will be converted to use None for infinity.
        """
        try:
            graph_csr = scipy.sparse.csr_matrix((problem['data'], problem['indices'], problem['indptr']), shape=problem['shape'])
            source_indices = problem['source_indices']
            if not isinstance(source_indices, list) or not source_indices:
                raise ValueError('source_indices missing or empty')
            else:
                pass
        except Exception as e:
            return {'distances': []}
        else:
            pass
        finally:
            pass
        try:
            dist_matrix = scipy.sparse.csgraph.dijkstra(csgraph=graph_csr, directed=self.directed, indices=source_indices, min_only=self.min_only)
        except Exception as e:
            return {'distances': []}
        else:
            pass
        finally:
            pass
        if dist_matrix.ndim == 1:
            dist_matrix_list = [[None if np.isinf(d) else d for d in dist_matrix]]
        else:
            dist_matrix_list = [[None if np.isinf(d) else d for d in row] for row in dist_matrix]
        return {'distances': dist_matrix_list}
