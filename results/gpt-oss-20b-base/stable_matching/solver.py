from collections import deque
from typing import Any, Dict, List


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        """
        Gale‑Shapley stable matching algorithm (proposer optimal).
        Accepts proposer_prefs and receiver_prefs either as lists of lists or
        dictionaries mapping indices to preference lists.
        Returns a dictionary with key "matching" and a list of receiver indices
        assigned to each proposer.
        """
        # Normalise preference inputs
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer_prefs: List[List[int]] = [prop_raw[i] for i in range(n)]
        else:
            proposer_prefs = list(prop_raw)
            n = len(proposer_prefs)

        if isinstance(recv_raw, dict):
            receiver_prefs = [recv_raw[i] for i in range(n)]
        else:
            receiver_prefs = list(recv_raw)

        # Precompute receiver ranking tables for O(1) preference comparison
        recv_rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(receiver_prefs):
            for rank, p in enumerate(prefs):
                recv_rank[r][p] = rank

        # Gale–Shapley main loop
        next_proposal = [0] * n          # next index in each proposer's list to propose
        recv_match = [None] * n          # current proposer matched to each receiver
        free_proposers = deque(range(n))  # proposers currently unmatched

        while free_proposers:
            p = free_proposers.popleft()
            r = proposer_prefs[p][next_proposal[p]]
            next_proposal[p] += 1

            current = recv_match[r]
            if current is None:
                recv_match[r] = p
            else:
                if recv_rank[r][p] < recv_rank[r][current]:
                    recv_match[r] = p
                    free_proposers.append(current)
                else:
                    free_proposers.append(p)

        # Build the matching list: matching[p] = r
        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r

        return {"matching": matching}
