from collections import deque
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        # ----- read preferences -----
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        # accept dict or list/tuple
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

        # ----- pre-compute ranking of proposers for each receiver -----
        recv_rank = [[0]*n for _ in range(n)]
        for r, prefs in enumerate(receiver_prefs):
            for rank, p in enumerate(prefs):
                recv_rank[r][p] = rank

        # ----- Gale–Shapley (proposer-optimal) -----
        next_prop = [0]*n
        recv_match: List[int | None] = [None]*n
        free = deque(range(n))  # faster pop from left

        while free:
            p = free.popleft()
            r = proposer_prefs[p][next_prop[p]]
            next_prop[p] += 1

            cur = recv_match[r]
            if cur is None:
                recv_match[r] = p
            else:
                # lower rank = more preferred
                if recv_rank[r][p] < recv_rank[r][cur]:
                    recv_match[r] = p
                    free.append(cur)
                else:
                    free.append(p)

        # ----- build proposer->receiver matching list -----
        matching = [0]*n
        for r, p in enumerate(recv_match):
            matching[p] = r

        return {"matching": matching}