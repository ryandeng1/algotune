import numpy as np
from typing import Any, List, Dict
import numba
from numba import njit

@njit
def find(parent, x):
    stack = []
    while parent[x] != x:
        stack.append(x)
        x = parent[x]
    root = x
    for node in stack:
        parent[node] = root
    return root

@njit
def union(parent, rank, x, y):
    rx = find(parent, x)
    ry = find(parent, y)
    if rx == ry:
        return False
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1
    return True

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        edges_arr = np.array(edges, dtype=np.float64)
        sorted_edges = edges_arr[edges_arr[:, 2].argsort()]
        
        parent = np.arange(num_nodes, dtype=np.int64)
        rank = np.zeros(num_nodes, dtype=np.int64)
        
        mst_edges = []
        for i in range(sorted_edges.shape[0]):
            u = int(sorted_edges[i, 0])
            v = int(sorted_edges[i, 1])
            w = float(sorted_edges[i, 2])
            if union(parent, rank, u, v):
                mst_edges.append([u, v, w])
        
        mst_edges = [[min(u, v), max(u, v), w] for u, v, w in mst_edges]
        mst_edges.sort(key=lambda x: (x[0], x[1]))
        
        return {"mst_edges": mst_edges}
