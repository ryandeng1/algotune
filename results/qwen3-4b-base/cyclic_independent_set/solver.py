import numba
import numpy as np
from numba import njit

class Solver:
    def solve(self, problem: tuple[int, int]) -> list[tuple[int, ...]]:
        num_nodes, n = problem
        assert num_nodes == 7, "Problem must be 7-node cycle"
        
        multipliers = np.array([7**(n-1-i) for i in range(n)], dtype=np.int64)
        
        @njit
        def priority(el):
            total = 0
            for i in range(n):
                x = el[i]
                if x > 4:
                    x = 4
                total += (x + 1) * multipliers[i]
            return total % 5
        
        total_candidates = 7**n
        candidates = np.zeros((total_candidates, n), dtype=np.int32)
        
        for i in range(total_candidates):
            for j in range(n):
                candidates[i, j] = (i // (7**(n-1-j))) % 7
        
        scores = np.zeros(total_candidates, dtype=np.float64)
        for i in range(total_candidates):
            scores[i] = priority(tuple(candidates[i, :]))
        
        selected = []
        for _ in range(total_candidates):
            idx = np.argmax(scores)
            selected.append(tuple(candidates[idx, :]))
            scores[idx] = -np.inf
        
        return selected
