import gzip
from typing import Any


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        plaintext = problem["plaintext"]
        compressed_data = gzip.compress(plaintext, compresslevel=9, mtime=0)
        return {"compressed_data": compressed_data}