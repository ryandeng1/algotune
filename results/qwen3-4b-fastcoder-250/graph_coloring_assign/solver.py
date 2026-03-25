import networkx as nx
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    G.add_edge(i, j)
        
        try:
            from networkx.algorithms.approximation import clique as approx_clique
            max_clique = approx_clique.max_clique(G)
            clique_size = len(max_clique)
        except Exception:
            clique_size = 1
        
        greedy_colors = nx.greedy_color(G, strategy="largest_first")
        greedy_count = len(set(greedy_colors.values()))
        
        if clique_size == greedy_count:
            return [v + 1 for v in greedy_colors.values()]
        
        model = cp_model.CpModel()
        H = greedy_count
        x = {}
        for i in range(n):
            for c in range(H):
                x[(i, c)] = model.NewBoolVar(f"x_{i}_{c}")
        
        w = {}
        for c in range(H):
            w[c] = model.NewBoolVar(f"w_{c}")
        
        for i in range(n):
            model.Add(sum(x[(i, c)] for c in range(H)) == 1)
        
        for u, v in G.edges():
            for c in range(H):
                model.Add(x[(u, c)] + x[(v, c)] <= w[c])
        
        for c in range(H):
            model.Add(w[c] <= sum(x[(i, c)] for i in range(n)))
        
        for c in range(1, H):
            model.Add(w[c-1] >= w[c])
        
        model.Minimize(sum(w[c] for c in range(H)))
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            return []
        
        color_assign = [0] * n
        for i in range(n):
            for c in range(H):
                if solver.Value(x[(i, c)]) == 1:
                    color_assign[i] = c + 1
                    break
        
        used_colors = sorted(set(color_assign))
        remap = {old: new for new, old in enumerate(used_colors, start=1)}
        return [remap[c] for c in color_assign]
