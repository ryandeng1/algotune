from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[int]]:
        # Extract proposals and acceptances
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        # Convert to list-of-lists if necessary
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer = [prop_raw[i] for i in range(n)]
        else:
            proposer = list(prop_raw)
            n = len(proposer)

        if isinstance(recv_raw, dict):
            receiver = [recv_raw[i] for i in range(n)]
        else:
            receiver = list(recv_raw)

        # Build receiver rank table
        recv_rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(receiver):
            rr = recv_rank[r]
            for rank, p in enumerate(prefs):
                rr[p] = rank

        # Gale–Shapley algorithm
        next_proposal = [0] * n
        recv_match = [-1] * n
        free = list(range(n))  # use LIFO stack for free proposers

        while free:
            p = free.pop()
            r = proposer[p][next_proposal[p]]
            next_proposal[p] += 1

            cur = recv_match[r]
            if cur == -1:
                recv_match[r] = p
            else:
                rr = recv_rank[r]
                if rr[p] < rr[cur]:
                    recv_match[r] = p
                    free.append(cur)   # previous match becomes free
                else:
                    free.append(p)     # current proposer remains free

        # Build matching list: matching[p] = r
        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r

        return {"matching": matching}