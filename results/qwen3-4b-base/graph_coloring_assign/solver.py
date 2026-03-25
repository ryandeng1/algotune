import networkx as nx
from networkx.algorithms.approximation import clique as approx_clique
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    G.add_edge(i, j)
        
        greedy_coloring = nx.greedy_color(G, strategy="largest_first")
        ub = len(set(greedy_coloring.values()))
        
        clique_set = approx_clique.max_clique(G)
        lb = len(clique_set)
        
        if lb == ub:
            return [greedy_coloring[i] + 1 for i in range(n)]
        
        model = cp_model.CpModel()
        
        x = {}
        for u in range(n):
            for i in range(ub):
                x[(u, i)] = model.NewBoolVar(f"x_{u}_{i}")
        
        for u in range(n):
            model.Add(sum(x[(u, i)] for i in range(ub)) == 1)
        
        for u, v in G.edges():
            for i in range(ub):
                model.Add(x[(u, i)] + x[(v, i)] <= 1)
        
        w = {}
        for i in range(ub):
            w[i] = model.NewBoolVar(f"w_{i}")
            model.Add(w[i] == sum(x[(u, i)] for u in range(n)))
        
        model.Minimize(sum(w[i] for i in range(ub)))
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            return []
        
        colors = [0] * n
        for u in range(n):
            for i in range(ub):
                if solver.Value(x[(u, i)]) == 1:
                    colors[u] = i + 1
                    break
        
        used_colors = sorted(set(colors))
        remap = {old: new for new, old in enumerate(used_colors, start=1)}
        return [remap[c] for c in colors]
