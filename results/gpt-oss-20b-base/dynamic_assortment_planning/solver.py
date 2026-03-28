from typing import Any

# Hungarian algorithm for maximum weight matching on a square matrix
def hungarian(weights):
    n = len(weights)
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = -weights[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment, -v[0]

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        T = problem['T']
        N = problem['N']
        prices = problem['prices']
        capacities = problem['capacities']
        probs = problem['probs']

        # Build slot list for each item
        slots = []
        item_of_slot = []
        for i in range(N):
            for _ in range(capacities[i]):
                slots.append(i)
                item_of_slot.append(i)

        M = len(slots)
        n = max(T, M)
        # construct weight matrix (n x n) padded with zeros
        weights = [[0] * n for _ in range(n)]
        for t in range(T):
            for s in range(M):
                i = slots[s]
                weights[t][s] = prices[i] * probs[t][i]
        # the rest of the matrix remains zero (dummy slots)

        assignment, _ = hungarian(weights)

        # Build result for each period
        offer = [-1] * T
        for t in range(T):
            s = assignment[t]
            if s < M:
                offer[t] = slots[s]
        return offer