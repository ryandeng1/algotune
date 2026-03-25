import itertools

class Solver:
    def solve(self, problem: list[list[float]]) -> list[int]:
        n = len(problem)
        if n <= 1:
            return [0, 0]
        
        best_max = float('inf')
        best_tour = None
        
        for perm in itertools.permutations(range(1, n)):
            tour = [0] + list(perm) + [0]
            max_edge = 0
            for i in range(len(tour) - 1):
                u = tour[i]
                v = tour[i+1]
                edge = problem[u][v]
                if edge > max_edge:
                    max_edge = edge
            if max_edge < best_max:
                best_max = max_edge
                best_tour = tour
        return best_tour
