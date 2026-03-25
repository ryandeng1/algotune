# solver.py
from collections import deque
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        """
        Gale–Shapley algorithm for the stable marriage problem.
        Returns a dictionary {"matching": [receiver_index_of_proposer_i, ...]}.
        The implementation is fully vectorised with integer lists and uses deque
        for O(1) pop/append operations.
        """
        # Normalise input to list of lists
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

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

        # Pre‑compute receiver ranking tables
        recv_rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(receiver_prefs):
            for rank, p in enumerate(prefs):
                recv_rank[r][p] = rank

        # Gale–Shapley core
        next_prop = [0] * n                 # next receiver index to propose to for each proposer
        recv_match = [None] * n             # current proposer matched to each receiver
        free = deque(range(n))              # proposers yet to be matched

        while free:
            p = free.popleft()
            r = proposer_prefs[p][next_prop[p]]
            next_prop[p] += 1

            cur = recv_match[r]
            if cur is None:
                recv_match[r] = p
            else:
                # decide if receiver prefers new proposer
                if recv_rank[r][p] < recv_rank[r][cur]:
                    recv_match[r] = p
                    free.append(cur)
                else:
                    free.append(p)

        # Build output matching array
        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r

        return {"matching": matching}
