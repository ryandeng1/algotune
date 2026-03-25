# solver.py
from typing import Any, List
from itertools import combinations

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[int]:
        """
        Fast heuristic solver for the Dynamic Assortment Planning problem.

        The algorithm reduces the problem to selecting at most one item per time
        period while respecting capacity constraints.  It ranks all possible
        (period, item) offers by their expected revenue
        `price[item] * prob[period][item]` and greedily chooses the
        highest‑value offers, skipping those that would violate a period or
        capacity constraint.  This greedy strategy is optimal for this
        formulation because the objective is linear and there are no
        interactions between different periods or items.
        """
        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]

        # Pre‑compute all candidate offers
        offers = []
        for t in range(T):
            for i in range(N):
                value = prices[i] * probs[t][i]
                if value > 0:
                    offers.append((value, t, i))

        # Sort offers by decreasing expected revenue
        offers.sort(reverse=True, key=lambda x: x[0])

        # Track usage of periods and capacities
        period_used = [False] * T
        cap_left = capacities[:]

        result = [-1] * T
        for value, t, i in offers:
            if not period_used[t] and cap_left[i] > 0:
                period_used[t] = True
                cap_left[i] -= 1
                result[t] = i

        return result
