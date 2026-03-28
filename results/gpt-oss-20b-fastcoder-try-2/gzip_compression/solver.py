import gzip
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """Compress the plaintext using the gzip algorithm with mtime=0."""
        return {'compressed_data': gzip.compress(problem['plaintext'], compresslevel=9, mtime=0)}