import networkx as nx

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]
        
        G = nx.DiGraph()
        source = 0
        sink = T + N + 1
        
        for t in range(T):
            G.add_edge(source, t + 1, capacity=1, cost=0)
        
        for t in range(T):
            for i in range(N):
                product_node = T + 1 + i
                cost = -prices[i] * probs[t][i]
                G.add_edge(t + 1, product_node, capacity=1, cost=cost)
        
        for i in range(N):
            product_node = T + 1 + i
            G.add_edge(product_node, sink, capacity=capacities[i], cost=0)
        
        flow_value, flow_cost, flow_dict = nx.min_cost_flow(G, source, sink)
        
        solution = [-1] * T
        for t in range(T):
            for i in range(N):
                product_node = T + 1 + i
                if flow_dict.get((t + 1, product_node), 0) == 1:
                    solution[t] = i
                    break
        return solution
