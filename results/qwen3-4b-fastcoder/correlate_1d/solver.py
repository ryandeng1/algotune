from typing import Any
from scipy import signal

class Solver:
    def solve(self, problem: list) -> list:
        mode = self.mode
        if mode == "valid":
            return [signal.correlate(a, b, mode=mode) for a, b in problem if b.shape[0] <= a.shape[0]]
        else:
            return [signal.correlate(a, b, mode=mode) for a, b in problem]