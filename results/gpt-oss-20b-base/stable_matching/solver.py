#!/usr/bin/env python3
# solver.py
from __future__ import annotations
from typing import Any, Dict, List
from collections import deque


class Solver:
    """
    Implements the Gale–Shapley stable matching algorithm.
    Input must contain:
      - proposer_prefs: list[tuple[int]] or dict[int, tuple[int]]
      - receiver_prefs: list[tuple[int]] or dict[int, tuple[int]]
    Returns a dict with key 'matching' whose value is a list where
    matching[p] is the index of the receiver matched to proposer p.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        # ----- parse preferences -------------------------------------------------
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        # Convert to list[tuple[int]] with 0‑based indices
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer_prefs: List[tuple[int]] = [tuple(prop_raw[i]) for i in range(n)]
        else:
            proposer_prefs = [tuple(p) for p in prop_raw]
            n = len(proposer_prefs)

        if isinstance(recv_raw, dict):
            receiver_prefs: List[tuple[int]] = [tuple(recv_raw[i]) for i in range(n)]
        else:
            receiver_prefs = [tuple(r) for r in recv_raw]

        # ----- build receiver rank matrix --------------------------------------
        # recv_rank[r][p] = position of proposer p in receiver r's preference list
        recv_rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(receiver_prefs):
            for rank, p in enumerate(prefs):
                recv_rank[r][p] = rank

        # ----- Gale–Shapley ----------------------------------------------------
        next_prop = [0] * n          # next receiver to propose to for each proposer
        recv_match: List[int | None] = [None] * n  # matched proposer for each receiver

        free = deque(range(n))      # stack of free proposers (LIFO for O(1) pop)

        while free:
            p = free.pop()
            r = proposer_prefs[p][next_prop[p]]
            next_prop[p] += 1

            cur = recv_match[r]
            if cur is None:
                recv_match[r] = p
            else:
                # If receiver prefers new proposer over current
                if recv_rank[r][p] < recv_rank[r][cur]:
                    recv_match[r] = p
                    free.append(cur)
                else:
                    free.append(p)

        # ----- construct final matching -----------------------------------------
        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r

        return {"matching": matching}