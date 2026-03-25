import numpy as np
from ortools.graph import min_cost_flow

class Solver:
    def solve(self, problem, **kwargs) -> list[list[float]]:
        capacity = problem["capacity"]
        cost = problem["cost"]
        s = problem["s"]
        t = problem["t"]
        n = len(capacity)
        
        flow_network = min_cost_flow.MinCostFlow()
        edges = []
        
        for i in range(n):
            for j in range(n):
                if capacity[i][j] > 0:
                    flow_network.AddArcWithCapacityAndCost(i, j, capacity[i][j], cost[i][j])
                    edges.append((i, j))
        
        flow_network.Solve(s, t)
        
        flow_matrix = [[0.0] * n for _ in range(n)]
        flow_values = flow_network.Flow()
        
        for idx, (i, j) in enumerate(edges):
            flow_matrix[i][j] = float(flow_values[idx])
        
        return flow_matrix
