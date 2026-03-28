class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        from scipy.linalg import expm
        A = problem['matrix']
        return {'exponential': expm(A).tolist()}