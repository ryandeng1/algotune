class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]
        
        dp = [[float('inf')] * n for _ in range(1 << n)]
        prev = [[-1] * n for _ in range(1 << n)]
        
        dp[1 << 0][0] = 0
        
        for mask in range(1 << n):
            for i in range(n):
                if dp[mask][i] == float('inf'):
                    continue
                if mask & (1 << i):
                    for j in range(n):
                        if not (mask & (1 << j)):
                            new_mask = mask | (1 << j)
                            new_cost = dp[mask][i] + problem[i][j]
                            if new_cost < dp[new_mask][j]:
                                dp[new_mask][j] = new_cost
                                prev[new_mask][j] = i
        
        full_mask = (1 << n) - 1
        best_city = -1
        best_cost = float('inf')
        for i in range(n):
            total_cost = dp[full_mask][i] + problem[i][0]
            if total_cost < best_cost:
                best_cost = total_cost
                best_city = i
        
        path = [best_city]
        current = best_city
        mask = full_mask
        for _ in range(n - 1):
            current = prev[mask][current]
            path.append(current)
            mask = mask & ~(1 << current)
        
        path.reverse()
        path.append(0)
        return path
