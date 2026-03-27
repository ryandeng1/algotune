from typing import Any

# Implementation of the Hungarian (Kuhn–Munkres) algorithm for the
# minimum‑weight assignment problem.
# Complexity: O(n^3).  Works for any square cost matrix (no infinity values).
def hungarian(cost):
    n = len(cost)
    matedial = list(range(n))
    perm = [-1] * n
    label_x = [max(row) for row in cost]
    label_y = [0] * n
    for x in range(n):
        slack = [float("inf")] * n
        slackx = [-1] * n
        prev = [-1] * n
        S = [False] * n
        T = [False] * n
        q = [x]
        prev[x] = -2
        cur = 0
        while True:
            i = q[cur]
            cur += 1
            S[i] = True
            for j in range(n):
                if T[j]:
                    continue
                cur_slack = label_x[i] + label_y[j] - cost[i][j]
                if cur_slack < slack[j]:
                    slack[j] = cur_slack
                    slackx[j] = i
            delta, min_j = min((slack[j], j) for j in range(n) if not T[j])
            for i in range(n):
                if S[i]:
                    label_x[i] -= delta
            for j in range(n):
                if T[j]:
                    label_y[j] += delta
                else:
                    slack[j] -= delta
            T[min_j] = True
            if perm[min_j] == -1:
                # augmenting path found
                j = min_j
                while True:
                    i = slackx[j]
                    temp = perm[j]
                    perm[j] = i
                    j = temp
                    if i == x:
                        break
                break
            else:
                q.append(perm[min_j])
                prev[perm[min_j]] = slackx[min_j]
    return perm  # perm[j] = i  meaning i matched to j


def solve(problem: dict[str, Any]) -> list[list[Any]]:
    """
    Solves the minimum‑weight assignment problem.
    Returns a 2‑D list containing the flow for each edge (adjacency matrix format).
    A flow of 1 indicates an assigned pair, 0 otherwise.
    """
    capacity = problem["capacity"]
    cost = problem["cost"]
    n = len(capacity)

    # Extract the usable cost matrix for edges that have positive capacity
    # An assignment can only use such edges; others are treated with a very large cost.
    LARGE = 10**9
    cost_matrix = [
        [
            cost[i][j] if capacity[i][j] > 0 else LARGE
            for j in range(n)
        ]
        for i in range(n)
    ]

    match = hungarian(cost_matrix)

    result = [[0] * n for _ in range(n)]
    for j, i in enumerate(match):
        if i != -1 and capacity[i][j] > 0:
            result[i][j] = 1

    return result