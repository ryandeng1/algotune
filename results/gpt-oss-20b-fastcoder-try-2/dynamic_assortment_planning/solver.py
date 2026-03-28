from typing import Any, List
import numpy as np
from scipy.optimize import linear_sum_assignment

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[int]:
        """
        Solve the DAP exactly with a weighted bipartite matching.
        For each time period we pick at most one offer, and each offer
        i can be chosen at most `capacities[i]` times across all periods.
        """
        T = problem['T']
        N = problem['N']
        prices = problem['prices']
        capacities = problem['capacities']
        probs = problem['probs']

        # total number of "slots" created by expanding capacities
        slots = sum(capacities)
        # Create matrix (T x slots) of weights, negative for lin_assign (min-cost)
        # If a slot belongs to offer i, weight = prices[i] * probs[t][i]
        cost = np.zeros((T, slots))
        idx = 0
        for i in range(N):
            for _ in range(capacities[i]):
                cost[:, idx] = -prices[i] * np.array(probs, dtype=np.float64)[:, i]
                idx += 1

        # Solve assignment problem (min-cost) on negative weights -> max profit
        row_ind, col_ind = linear_sum_assignment(cost)

        # Build result: for each period t, find chosen offer or -1
        offer = [-1] * T
        # Map slot index back to offer id
        slot_to_offer = []
        for i in range(N):
            slot_to_offer.extend([i] * capacities[i])

        for r, c in zip(row_ind, col_ind):
            if r < T:  # accounts for possible extra columns
                offer[r] = slot_to_offer[c]

        return offer