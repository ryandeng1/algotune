from typing import Any
from scipy import signal

class Solver:
    def solve(self, problem: list) -> list:
        return [signal.upfirdn(h, x, up=up, down=down) for h, x, up, down in problem]