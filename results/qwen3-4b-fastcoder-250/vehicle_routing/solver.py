from ortools.routing import RoutingIndexManager, RoutingModel

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)
        
        manager = RoutingIndexManager(n, K, depot)
        routing_model = RoutingModel(manager)
        routing_model.SetArcCosts(D)
        
        solution = routing_model.Solve()
        
        routes = []
        for vehicle_id in range(K):
            route = [depot]
            current = depot
            while True:
                next_node = solution.GetNextNode(vehicle_id, current)
                if next_node == depot:
                    break
                route.append(next_node)
                current = next_node
            routes.append(route)
        
        return routes
