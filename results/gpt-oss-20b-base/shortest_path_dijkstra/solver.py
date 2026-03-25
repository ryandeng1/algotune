import numpy as np
import scipy.sparse
import scipy.sparse.csgraph


class Solver:
    """Fast all‑pairs shortest path solver for sparse undirected weighted graphs.

    Uses :func:`scipy.sparse.csgraph.shortest_path` with Dijkstra's algorithm,
    which is optimised for sparse graphs and outperforms naïve Python
    implementations on all but extremely small instances.
    """

    def __init__(self) -> None:
        # Parameters for shortest_path: undirected, Dijkstra method
        self.directed: bool = False
        self.method: str = "D"  # 'C' for Dijkstra, 'LL' for repeated Floyd‑Warshall, etc.

    def solve(self, problem: dict[str, any]) -> dict[str, list[list[float]]]:
        """
        Parameters
        ----------
        problem : dict
            CSR representation of a weighted undirected graph:
                * data   : list[float]
                * indices: list[int]
                * indptr : list[int]
                * shape  : list[int | tuple] of length 2 (n, n)

        Returns
        -------
        dict
            ``{"distance_matrix": ...}`` where the value is a list of lists
            of floats, with ``None`` representing infinity (no path).
        """
        try:
            mat = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
        except Exception:
            return {"distance_matrix": []}

        try:
            dist = scipy.sparse.csgraph.shortest_path(
                mat,
                directed=self.directed,
                method=self.method,
            )
        except Exception:
            return {"distance_matrix": []}

        # Convert infinities to None for JSON serialisation
        # Numpy operations are vectorised and fast
        inf_mask = np.isinf(dist)
        if inf_mask.any():
            result = dist.tolist()
            for i in range(dist.shape[0]):
                row = result[i]
                for j in range(dist.shape[1]):
                    if inf_mask[i, j]:
                        row[j] = None
        else:
            result = dist.tolist()

        return {"distance_matrix": result}
