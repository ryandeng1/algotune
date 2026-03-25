class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        n = problem["num_nodes"]
        edges_g1 = problem["edges_g1"]
        edges_g2 = problem["edges_g2"]
        
        deg_g1 = [0] * n
        deg_g2 = [0] * n
        
        for u, v in edges_g1:
            deg_g1[u] += 1
            deg_g1[v] += 1
            
        for u, v in edges_g2:
            deg_g2[u] += 1
            deg_g2[v] += 1
            
        from collections import defaultdict
        groups_g1 = defaultdict(list)
        groups_g2 = defaultdict(list)
        
        for i in range(n):
            groups_g1[deg_g1[i]].append(i)
            groups_g2[deg_g2[i]].append(i)
        
        mapping = [0] * n
        for degree in sorted(groups_g1.keys()):
            g1_list = groups_g1[degree]
            g2_list = groups_g2[degree]
            for idx, node in enumerate(g1_list):
                mapping[node] = g2_list[idx]
        
        return {"mapping": mapping}
