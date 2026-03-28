from typing import Any, List

def solve(problem: dict[str, Any]) -> List[List[int]]:
    """
    Very fast heuristic solution for the VRP.
    Every vehicle visits one customer (if possible).
    If there are fewer customers than vehicles, some vehicles stay at the depot.
    """
    D = problem['D']
    K = problem['K']
    depot = problem['depot']

    n = len(D)          # total nodes (including depot)
    customers = [i for i in range(n) if i != depot]

    routes: List[List[int]] = []

    # Assign one customer to each vehicle until we run out of customers.
    for i, cust in enumerate(customers):
        if i < K:
            routes.append([depot, cust, depot])
        else:
            # Remaining vehicles stay at the depot
            break

    # If there are more vehicles than customers, add empty routes.
    while len(routes) < K:
        routes.append([depot, depot])

    return routes