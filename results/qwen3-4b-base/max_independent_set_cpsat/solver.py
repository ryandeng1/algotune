import collections
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        adj_list = []
        for i in range(n):
            adj_list.append([j for j in range(n) if problem[i][j] == 1])
        
        color = [-1] * n
        is_bipartite = True
        for i in range(n):
            if color[i] == -1:
                queue = collections.deque([i])
                color[i] = 0
                while queue and is_bipartite:
                    u = queue.popleft()
                    for v in adj_list[u]:
                        if color[v] == -1:
                            color[v] = color[u] ^ 1
                            queue.append(v)
                        elif color[v] == color[u]:
                            is_bipartite = False
                            break
                if not is_bipartite:
                    break
        if is_bipartite:
            partition0 = [i for i in range(n) if color[i] == 0]
            partition1 = [i for i in range(n) if color[i] == 1]
            return partition0 if len(partition0) >= len(partition1) else partition1
        else:
            model = cp_model.CpModel()
            nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if problem[i][j] == 1:
                        model.Add(nodes[i] + nodes[j] <= 1)
            model.Maximize(sum(nodes))
            solver = cp_model.CpSolver()
            status = solver.Solve(model)
            if status == cp_model.OPTIMAL:
                selected = [i for i in range(n) if solver.Value(nodes[i]) == 1]
                return selected
            else:
                return []
