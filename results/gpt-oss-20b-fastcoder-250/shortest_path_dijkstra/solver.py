import numpy as np
from typing import Any, Dict, List, Union

class Solver:
    def solve(self, problem: Dict[str, Union[List[float], List[int], List[int], List[int]]]) -> Dict[str, List[List[Union[float, None]]]]:
        """
        Compute all-pairs shortest paths on an undirected weighted graph given in CSR format.
        Uses the Floyd‑Warshall algorithm on a dense adjacency matrix, which is fast for
        reasonably sized graphs. Unreachable pairs are represented as None in the output.
        """
        data = problem["data"]
        indices = problem["indices"]
        indptr = problem["indptr"]
        n = problem["shape"][0]

        # Build dense adjacency matrix with infinities
        inf = np.inf
        dist = np.full((n, n), inf, dtype=float)
        np.fill_diagonal(dist, 0.0)

        # Fill edges (undirected)
        for src in range(n):
            start, end = indptr[src], indptr[src + 1]
            for idx in range(start, end):
                dst = indices[idx]
                w = data[idx]
                # store minimal weight for multiple edges
                if w < dist[src, dst]:
                    dist[src, dst] = w
                    dist[dst, src] = w

        # Floyd‑Warshall: O(n^3) with vectorized operations
        for k in range(n):
            # Local copy to avoid repeated indexing in Python loop
            dist_k = dist[:, k][:, None]
            alt = dist_k + dist[k, :]
            # Element‑wise min with broadcasting
            np.minimum(dist, alt, out=dist)

        # Convert infinities to None for JSON serialization
        result = [[None if np.isinf(d) else float(d) for d in row] for row in dist]
        return {"distance_matrix": result}
