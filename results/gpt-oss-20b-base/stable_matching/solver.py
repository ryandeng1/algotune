from collections import deque
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        # ---- read preferences --------------------------------------------------
        prop = problem["proposer_prefs"]
        recv = problem["receiver_prefs"]

        # ensure list-of-lists format
        if isinstance(prop, dict):
            n = len(prop)
            proposer_prefs = [prop[i] for i in range(n)]
        else:
            proposer_prefs = list(prop)
            n = len(proposer_prefs)

        if isinstance(recv, dict):
            receiver_prefs = [recv[i] for i in range(n)]
        else:
            receiver_prefs = list(recv)

        # ---- create rank matrix for receivers -------------------------------
        recv_rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(receiver_prefs):
            for idx, p in enumerate(prefs):
                recv_rank[r][p] = idx

        # ---- Gale–Shapley (proposer‑optimal) --------------------------------
        next_prop = [0] * n                 # next receiver to propose to
        recv_match = [None] * n             # matched proposer for each receiver
        free = deque(range(n))              # proposers who are free

        while free:
            p = free.popleft()
            r = proposer_prefs[p][next_prop[p]]
            next_prop[p] += 1

            cur = recv_match[r]
            if cur is None:
                recv_match[r] = p
            else:
                if recv_rank[r][p] < recv_rank[r][cur]:
                    recv_match[r] = p
                    free.append(cur)
                else:
                    free.append(p)

        # ---- convert to proposer‑to‑receiver mapping -------------------------
        matching = [0] * n
        for r, p in enumerate(recv_match):
            if p is not None:
                matching[p] = r

        return {"matching": matching}