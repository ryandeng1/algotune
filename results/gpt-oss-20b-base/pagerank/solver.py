def solve(problem: dict[str, list[list[int]]]) -> dict[str, list[float]]:
    # PageRank implementation (power iteration, without networkx)
    adj = problem['adjacency_list']
    n = len(adj)
    if n == 0:
        return {'pagerank_scores': []}
    if n == 1:
        return {'pagerank_scores': [1.0]}

    # Build out‑degree and incoming edges for each node
    out_deg = [len(lst) for lst in adj]
    incoming = [[] for _ in range(n)]
    for u, neigh in enumerate(adj):
        for v in neigh:
            incoming[v].append(u)

    alpha = getattr(solve, "alpha", 0.85)   # damping factor
    max_iter = getattr(solve, "max_iter", 100)
    tol = getattr(solve, "tol", 1e-6)

    # Initial rank vector (uniform)
    rank = [1.0 / n] * n
    teleport = (1.0 - alpha) / n

    for _ in range(max_iter):
        new_rank = [teleport] * n
        # Expected score from incoming edges
        for v in range(n):
            s = 0.0
            for u in incoming[v]:
                s += rank[u] / out_deg[u] if out_deg[u] else 0.0
            new_rank[v] += alpha * s
        # Check convergence
        diff = sum(abs(new_rank[i] - rank[i]) for i in range(n))
        if diff < tol:
            break
        rank = new_rank

    # Ensure each score is a Python float
    return {'pagerank_scores': [float(r) for r in rank]}