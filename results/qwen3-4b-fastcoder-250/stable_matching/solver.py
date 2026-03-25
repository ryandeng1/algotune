import numpy as np
from collections import deque

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        proposer_prefs = problem["proposer_prefs"]
        receiver_prefs = problem["receiver_prefs"]
        n = len(proposer_prefs)
        
        proposer_prefs = np.array(proposer_prefs, dtype=int)
        receiver_prefs = np.array(receiver_prefs, dtype=int)
        
        recv_rank = np.zeros((n, n), dtype=int)
        for r in range(n):
            for rank, p in enumerate(receiver_prefs[r]):
                recv_rank[r, p] = rank
        
        next_prop = np.zeros(n, dtype=int)
        recv_match = np.full(n, -1, dtype=int)
        free = deque(range(n))
        
        while free:
            p = free.popleft()
            r = proposer_prefs[p, next_prop[p]]
            next_prop[p] += 1
            
            cur = recv_match[r]
            if cur == -1:
                recv_match[r] = p
            else:
                if recv_rank[r, p] < recv_rank[r, cur]:
                    recv_match[r] = p
                    free.append(cur)
                else:
                    free.append(p)
        
        matching = np.zeros(n, dtype=int)
        for r in range(n):
            p = recv_match[r]
            if p != -1:
                matching[p] = r
        
        return {"matching": matching.tolist()}
