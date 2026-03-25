class Solver:
    def solve(self, problem: list[list[float]]) -> list[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append(problem[i][j])
        
        if not edges:
            return [0, 0]
        
        sorted_edges = sorted(set(edges))
        
        for x in sorted_edges:
            if self.is_hamiltonian(x, problem, n):
                return self.find_tour(x, problem, n)
        
        return self.find_tour(sorted_edges[-1], problem, n)
    
    def is_hamiltonian(self, x: float, problem: list[list[float]], n: int) -> bool:
        dp = [[False] * n for _ in range(1 << n)]
        dp[1 << 0][0] = True
        
        for mask in range(1 << n):
            for i in range(n):
                if not dp[mask][i]:
                    continue
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    if problem[i][j] <= x:
                        new_mask = mask | (1 << j)
                        dp[new_mask][j] = True
        
        full_mask = (1 << n) - 1
        for i in range(n):
            if dp[full_mask][i] and problem[i][0] <= x:
                return True
        return False
    
    def find_tour(self, x: float, problem: list[list[float]], n: int) -> list[int]:
        dp = [[False] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        dp[1 << 0][0] = True
        
        for mask in range(1 << n):
            for i in range(n):
                if not dp[mask][i]:
                    continue
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    if problem[i][j] <= x:
                        new_mask = mask | (1 << j)
                        if not dp[new_mask][j]:
                            dp[new_mask][j] = True
                            parent[new_mask][j] = i
        
        full_mask = (1 << n) - 1
        for i in range(n):
            if dp[full_mask][i] and problem[i][0] <= x:
                path = [i]
                cur = i
                mask = full_mask
                while mask != (1 << 0):
                    cur = parent[mask][cur]
                    path.append(cur)
                    mask &= ~(1 << cur)
                path.reverse()
                return path + [0]
        
        return [0] * (n + 1)
