class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]
        
        dp = [[float('inf')] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        
        dp[1][0] = 0
        
        for mask in range(1 << n):
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if not (mask & (1 << j)):
                        new_mask = mask | (1 << j)
                        new_cost = dp[mask][i] + problem[i][j]
                        if new_cost < dp[new_mask][j]:
                            dp[new_mask][j] = new_cost
                            parent[new_mask][j] = i
        
        full_mask = (1 << n) - 1
        min_cost = float('inf')
        best_i = -1
        for i in range(n):
            if dp[full_mask][i] + problem[i][0] < min_cost:
                min_cost = dp[full_mask][i] + problem[i][0]
                best_i = i
        
        path = []
        current = best_i
        while current != 0:
            path.append(current)
            current = parent[full_mask][current]
        path.append(0)
        path.reverse()
        path.append(0)
        return path
