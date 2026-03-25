import numpy as np
from ortools.routing import RoutingModel

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)
        
        manager = RoutingModel(n, K, depot)
        manager.SetArcCosts(D)
        solution = manager.Solve()
        
        routes = []
        for vehicle_id in range(K):
            route = [depot]
            current = solution.GetVehicleStartNode(vehicle_id)
            while current != depot:
                next_node = solution.GetNextNode(current)
                route.append(next_node)
                current = next_node
            route.append(depot)
            routes.append(route)
        return routes
