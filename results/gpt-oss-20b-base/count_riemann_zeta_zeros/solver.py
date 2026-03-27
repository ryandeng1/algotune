from mpmath import mp


class Solver:
    def solve(self, problem: dict) -> dict:
        """Count zeta zeros along the critical strip with imaginary part ≤ t using mpmath."""
        t = problem.get("t")
        if t is None:
            raise ValueError("Input dictionary must contain key 't'")
        # Directly use mp.nzeros for the count
        return {"result": mp.nzeros(t)}