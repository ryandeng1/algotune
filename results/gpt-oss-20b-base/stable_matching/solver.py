from collections import deque
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        # Normalise input: list of lists
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer_prefs: List[List[int]] = [prop_raw[i] for i in range(n)]
        else:
            proposer_prefs = list(prop_raw)
            n = len(proposer_prefs)

        if isinstance(recv_raw, dict):
            receiver_prefs: List[List[int]] = [recv_raw[i] for i in range(n)]
        else:
            receiver_prefs = list(recv_raw)

        # rank[r][p] = position of proposer p in receiver r's preference list
        rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(receiver_prefs):
            for pos, p in enumerate(prefs):
                rank[r][p] = pos

        next_prop = [0] * n          # next receiver index to propose to
        recv_match = [None] * n      # current match for each receiver

        free = deque(range(n))      # queue of free proposers

        while free:
            p = free.popleft()
            r = proposer_prefs[p][next_prop[p]]
            next_prop[p] += 1
            cur = recv_match[r]
            if cur is None:
                recv_match[r] = p
            elif rank[r][p] < rank[r][cur]:
                recv_match[r] = p
                free.append(cur)
            else:
                free.append(p)

        # Convert receiver->proposer matches to proposer->receiver list
        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r

        return {"matching": matching}