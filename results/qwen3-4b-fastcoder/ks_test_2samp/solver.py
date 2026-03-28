class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        sample1 = problem["sample1"]
        sample2 = problem["sample2"]
        result = stats.ks_2samp(sample1, sample2)
        return {"statistic": result.statistic, "pvalue": result.pvalue}