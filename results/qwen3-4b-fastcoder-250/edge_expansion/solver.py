class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, float]:
        adj_list = problem["adjacency_list"]
        nodes_S = problem["nodes_S"]
        n = len(adj_list)
        
        if len(nodes_S) == 0 or len(nodes_S) == n:
            return {"edge_expansion": 0.0}
        
        set_S = set(nodes_S)
        count_out = 0
        for node in nodes_S:
            for neighbor in adj_list[node]:
                if neighbor not in set_S:
                    count_out += 1
        
        expansion = count_out / len(nodes_S)
        return {"edge_expansion": expansion}
