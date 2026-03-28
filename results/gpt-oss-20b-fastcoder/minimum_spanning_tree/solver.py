from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Compute the Minimum Spanning Tree (MST) of an undirected weighted graph
        using a vectorized Kruskal algorithm (union‑find).  This implementation
        avoids the heavy NetworkX machinery and therefore runs much faster.
        """
        num_nodes = problem['num_nodes']
        edges = np.array(problem['edges'], dtype=np.float64)

        # separate columns: u, v, w
        u = edges[:, 0].astype(np.int32)
        v = edges[:, 1].astype(np.int32)
        w = edges[:, 2]

        # sort edges by weight
        idx = np.argsort(w)
        u = u[idx]
        v = v[idx]
        w = w[idx]

        parent = np.arange(num_nodes, dtype=np.int32)
        rank = np.zeros(num_nodes, dtype=np.int32)

        def find(x: np.int32) -> np.int32:
            # path compression
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        mst = []
        for ui, vi, wi in zip(u, v, w):
            pu = find(ui)
            pv = find(vi)
            if pu != pv:
                # union by rank
                if rank[pu] < rank[pv]:
                    parent[pu] = pv
                elif rank[pu] > rank[pv]:
                    parent[pv] = pu
                else:
                    parent[pv] = pu
                    rank[pu] += 1
                # normalize order (u <= v)
                if ui > vi:
                    ui, vi = vi, ui
                mst.append([ui, vi, float(wi)])

        mst.sort(key=lambda x: (x[0], x[1]))
        return {'mst_edges': mst}