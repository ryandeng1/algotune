import gzip
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """Compress plaintext using gzip with mtime=0."""
        # Directly return the compressed bytes without exception handling overhead
        return {'compressed_data': gzip.compress(problem['plaintext'], compresslevel=9, mtime=0)}