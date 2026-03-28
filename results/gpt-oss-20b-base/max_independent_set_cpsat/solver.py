# fast maximum independent set solver using bit‑set backtracking
# (simple branch&bound, works fast for graphs up to ~60 nodes)

from typing import List, Tuple

class Solver:

    def solve(self, problem: List[List[int]]) -> List[int]:
        n = len(problem)

        # adjacency as bit masks
        neighbors = [0] * n
        for i in range(n):
            mask = 0
            row = problem[i]
            for j, val in enumerate(row):
                if val:
                    mask |= 1 << j
            neighbors[i] = mask

        # ordering of vertices (deg heuristic)
        order = sorted(range(n), key=lambda x: bin(neighbors[x]).count("1"))

        best_set: int = 0
        best_size: int = 0

        def dfs(idx: int, cur_set: int, cur_size: int, cand: int):
            nonlocal best_set, best_size
            if idx == n:
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = cur_set
                return
            # bound
            remaining = bin(cand).count("1")
            if cur_size + remaining <= best_size:
                return
            v = order[idx]
            if cand >> v & 1:
                # take v
                dfs(idx + 1,
                    cur_set | (1 << v),
                    cur_size + 1,
                    cand & ~neighbors[v] & ~(1 << v))
                # skip v
                dfs(idx + 1,
                    cur_set,
                    cur_size,
                    cand & ~(1 << v))

        dfs(0, 0, 0, (1 << n) - 1)

        # convert bitset to list of indices
        result = [i for i in range(n) if best_set >> i & 1]
        return result