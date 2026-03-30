from collections import deque

class Solver:
    """Fast implementation of the Stable Marriage Problem
    using the Gale–Shapley algorithm.
    """

    def solve(self, problem: dict) -> dict[str, list[int]]:
        # Unpack and normalise preference lists
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

        # Build receiver ranking matrix
        recv_rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(receiver_prefs):
            for rank, p in enumerate(prefs):
                recv_rank[r][p] = rank

        # Gale–Shapley algorithm
        next_prop = [0] * n          # index of next receiver to propose to
        recv_match = [None] * n      # current match of each receiver
        free_proposers = deque(range(n))

        while free_proposers:
            p = free_proposers.popleft()
            r = proposer_prefs[p][next_prop[p]]
            next_prop[p] += 1

            cur = recv_match[r]
            if cur is None:
                recv_match[r] = p
            else:
                # Lower rank means higher preference
                if recv_rank[r][p] < recv_rank[r][cur]:
                    recv_match[r] = p
                    free_proposers.append(cur)
                else:
                    free_proposers.append(p)

        # Convert receiver -> proposer map to proposer -> receiver
        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r

        return {"matching": matching}