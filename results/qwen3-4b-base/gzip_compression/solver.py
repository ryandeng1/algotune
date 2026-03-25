import gzip
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        plaintext = problem["plaintext"]
        compressed_data = gzip.compress(plaintext, compresslevel=9, mtime=0)
        return {"compressed_data": compressed_data}
