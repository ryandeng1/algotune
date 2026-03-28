from collections import deque
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        # unpack problem
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        # ensure list format
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer_prefs = [prop_raw[i] for i in range(n)]
        else:
            proposer_prefs = list(prop_raw)
            n = len(proposer_prefs)

        if isinstance(recv_raw, dict):
            receiver_prefs = [recv_raw[i] for i in range(n)]
        else:
            receiver_prefs = list(recv_raw)

        # build receiver rank tables
        recv_rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(receiver_prefs):
            rank_row = recv_rank[r]
            for rank, p in enumerate(prefs):
                rank_row[p] = rank

        # Gale–Shapley (proposer‑optimal) in a single pass
        next_prop = [0] * n
        recv_match = [None] * n
        free = deque(range(n))

        while free:
            p = free.popleft()
            r = proposer_prefs[p][next_prop[p]]
            next_prop[p] += 1
            cur = recv_match[r]
            if cur is None:
                recv_match[r] = p
            else:
                rank_row = recv_rank[r]
                if rank_row[p] < rank_row[cur]:
                    recv_match[r] = p
                    free.append(cur)
                else:
                    free.append(p)

        # convert receiver‑centric match to proposer‑centric list
        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r

        return {"matching": matching}