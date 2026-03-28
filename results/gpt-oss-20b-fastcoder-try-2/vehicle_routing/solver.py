from typing import Any, List

def solve(problem: dict[str, Any]) -> List[List[int]]:
    """
    Fast greedy nearest‑neighbor solver for the VRP.

    :param problem: Dictionary containing:
        - "D": cost matrix (list of lists)
        - "K": number of vehicles
        - "depot": depot node index
    :return: List of K routes, each starting and ending at the depot.
    """
    D = problem["D"]
    K = problem["K"]
    depot = problem["depot"]
    n = len(D)

    # Build a list of all nodes except the depot
    nodes = [i for i in range(n) if i != depot]
    remaining = set(nodes)

    routes = []
    for _ in range(K):
        if not remaining:
            routes.append([depot, depot])  # empty route for unused vehicles
            continue

        # start from the depot, pick the nearest unvisited node
        current = depot
        route = [depot]
        # choose an initial unvisited node that is closest to the depot
        next_node = min(remaining, key=lambda v: D[current][v])
        current = next_node
        route.append(current)
        remaining.remove(current)

        # continue until all nodes are visited
        while remaining:
            # move to the nearest unvisited node
            next_node = min(remaining, key=lambda v: D[current][v])
            current = next_node
            route.append(current)
            remaining.remove(current)

        route.append(depot)
        routes.append(route)

    # for any remaining vehicles, return a trivial route
    for _ in range(K - len(routes)):
        routes.append([depot, depot])

    return routes