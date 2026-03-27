from collections import deque
from typing import Any, List, Dict


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        prop_raw, recv_raw = problem["proposer_prefs"], problem["receiver_prefs"]

        # normalize to list‑of‑lists (indexable) and determine size
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

        # build ranking tables for receivers
        recv_rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(receiver_prefs):
            for rank, p in enumerate(prefs):
                recv_rank[r][p] = rank

        # classic Gale‑Shapley
        next_prop = [0] * n                # next proposal index for each proposer
        recv_match = [None] * n            # current match of each receiver
        free = deque(range(n))             # pool of free proposers

        while free:
            p = free.popleft()
            r = proposer_prefs[p][next_prop[p]]
            next_prop[p] += 1

            cur = recv_match[r]
            if cur is None:
                recv_match[r] = p
            elif recv_rank[r][p] < recv_rank[r][cur]:
                recv_match[r] = p
                free.append(cur)
            else:
                free.append(p)

        # convert receiver→proposer mapping to proposer→receiver mapping
        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r

        return {"matching": matching}