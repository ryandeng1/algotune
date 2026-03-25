#!/usr/bin/env python3
# solver.py
import math
from typing import Any, Dict, List

import numpy as np
from scipy.linalg import expm


class Solver:
    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Compute communicability matrix e^A for an undirected graph given by an adjacency list.
        The result is returned as a nested dictionary with integer keys.
        """
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)

        if n == 0:
            return {"communicability": {}}

        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for i, neighbors in enumerate(adj_list):
            for j in neighbors:
                A[i, j] = 1.0
                A[j, i] = 1.0

        # Compute matrix exponential
        ExpA = expm(A)

        # Convert to dictionary of dictionaries
        comm: Dict[int, Dict[int, float]] = {}
        for i in range(n):
            row_dict: Dict[int, float] = {}
            for j in range(n):
                val = ExpA[i, j]
                # Ensure finiteness
                if not math.isfinite(val):
                    val = float("nan")
                row_dict[j] = float(val)
            comm[i] = row_dict

        return {"communicability": comm}
