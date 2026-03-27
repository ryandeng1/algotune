from collections import deque
from typing import Any, Dict, List

def solve(problem: Dict[str, Any]) -> Dict[str, List[int]]:
    prop_raw = problem["proposer_prefs"]
    recv_raw = problem["receiver_prefs"]

    # Normalise preferences to list‑of‑lists
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

    # Build receivers' ranking tables for O(1) comparison
    recv_rank = [[0] * n for _ in range(n)]
    for r, prefs in enumerate(receiver_prefs):
        for rank, p in enumerate(prefs):
            recv_rank[r][p] = rank

    # Gale‑Shapley
    next_prop = [0] * n            # next proposal index for each proposer
    recv_match = [None] * n        # current match of each receiver
    free = deque(range(n))         # proposers that are free

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

    # Convert receiver→proposer matches to proposer→receiver
    matching = [0] * n
    for r, p in enumerate(recv_match):
        matching[p] = r

    return {"matching": matching}