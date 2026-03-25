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
            G.add_edge(source, t + 1, capacity=1, weight=0)
        
        for t in range(T):
            for i in range(N):
                weight = -prices[i] * probs[t][i]
                G.add_edge(t + 1, T + 1 + i, capacity=1, weight=weight)
        
        for i in range(N):
            G.add_edge(T + 1 + i, sink, capacity=capacities[i], weight=0)
        
        flow = nx.min_cost_flow(G, source, sink)
        
        offer = [-1] * T
        for t in range(T):
            for i in range(N):
                edge = (t + 1, T + 1 + i)
                if edge in flow and flow[edge] == 1.0:
                    offer[t] = i
                    break
        return offer
