class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, float]:
        adj_list = problem["adjacency_list"]
        nodes_S_list = problem["nodes_S"]
        n = len(adj_list)
        
        if n == 0 or not nodes_S_list:
            return {"edge_expansion": 0.0}
        if len(nodes_S_list) == n:
            return {"edge_expansion": 0.0}
        
        S_set = set(nodes_S_list)
        size_S = len(nodes_S_list)
        count_out = 0
        
        for u in nodes_S_list:
            for v in adj_list[u]:
                if v not in S_set:
                    count_out += 1
        
        return {"edge_expansion": count_out / size_S}
