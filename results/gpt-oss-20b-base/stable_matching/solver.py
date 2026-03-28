from collections import deque
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        # read prefs lists (already 0‑based indices)
        proposers = problem["proposer_prefs"]
        receivers = problem["receiver_prefs"]
        if isinstance(proposers, dict):
            n = len(proposers)
            proposer_prefs = [proposers[i] for i in range(n)]
        else:
            proposer_prefs = list(proposers)
            n = len(proposer_prefs)
        if isinstance(receivers, dict):
            receiver_prefs = [receivers[i] for i in range(n)]
        else:
            receiver_prefs = list(receivers)

        # rank matrix for receivers
        recv_rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(receiver_prefs):
            for rank, p in enumerate(prefs):
                recv_rank[r][p] = rank

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
                pr = recv_rank[r][p]
                cr = recv_rank[r][cur]
                if pr < cr:
                    recv_match[r] = p
                    free.append(cur)
                else:
                    free.append(p)

        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r
        return {"matching": matching}