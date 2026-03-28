from collections import deque
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        # Unpack preferences (they may be dicts or lists)
        prop_raw = problem['proposer_prefs']
        recv_raw = problem['receiver_prefs']
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            props = [prop_raw[i] for i in range(n)]
        else:
            props = list(prop_raw)
            n = len(props)
        if isinstance(recv_raw, dict):
            recvs = [recv_raw[i] for i in range(n)]
        else:
            recvs = list(recv_raw)

        # Build rank map for receivers
        recv_rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(recvs):
            ranks = recv_rank[r]
            for rank, p in enumerate(prefs):
                ranks[p] = rank

        # Gale–Shapley (proposer-optimal)
        next_prop = [0] * n                # next receiver index to propose for each proposer
        recv_match = [None] * n            # current proposer matched to each receiver
        free = deque(range(n))             # proposers that are free

        while free:
            p = free.popleft()
            r = props[p][next_prop[p]]
            next_prop[p] += 1
            cur = recv_match[r]
            if cur is None:
                recv_match[r] = p
            else:
                r_ranks = recv_rank[r]
                if r_ranks[p] < r_ranks[cur]:
                    recv_match[r] = p
                    free.append(cur)
                else:
                    free.append(p)

        # Build matching list: proposer -> receiver
        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r

        return {'matching': matching}