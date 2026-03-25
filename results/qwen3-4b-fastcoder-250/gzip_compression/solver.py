import gzip
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        plaintext = problem["plaintext"]
        return {"compressed_data": gzip.compress(plaintext, compresslevel=9, mtime=0)}
